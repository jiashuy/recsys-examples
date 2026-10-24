/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/

// ---------------------------------------------------------------------------
// Chained bucket table: the kernels. See docs/chained_bucket_index_design.md.
//
// Several tables are served by one launch: each key carries a table id, and a
// device array of TableView structs (one per table) holds the pointers and sizes.
//
// A table is `num_buckets` buckets of `cap` slots. A key belongs to bucket
// hash(key) % num_buckets and, inside it, to chain (hash >> 32) % cap. A bucket
// has `cap` chain heads, one per slot, and every slot has a `next`; both are
// bucket-local offsets (uint16, 0xFFFF for none), so a chain never leaves its
// bucket. The occupied slots of a bucket are always the prefix [0, bucket_size).
// A slot holds (key, score, index): `index` is the row of the key's value, taken
// from a pool of rows that is independent of where the key sits.
//
// Two operations are here; the rest (sorting a step's keys by bucket, the row
// pool, the segment arithmetic) is done on the Python side with torch ops.
//
//   find   one thread per key: walk the key's chain, return its row, and for a
//          hit optionally update its score. Keys are unique, so no two threads
//          write the same slot.
//   place  one warp per bucket segment: put the bucket's new keys in its free
//          slots and, if there are more new keys than free slots, evict the
//          lowest-scored members to make room. Everything that touches a bucket
//          is done by that bucket's warp, so no atomics are needed.
// ---------------------------------------------------------------------------

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cstdint>
#include <tuple>
#include <vector>

#include "embedding_cache/guard.h"

namespace py = pybind11;

namespace dyn_emb {
namespace chained {

constexpr uint16_t kNull = 0xFFFF;
constexpr int kMaxBucketCapacity = 256;
constexpr int kWarpsPerBlock = 4;

__host__ __device__ __forceinline__ uint64_t hash64(int64_t key) {
  uint64_t x = static_cast<uint64_t>(key);
  x ^= x >> 33;
  x *= 0xff51afd7ed558ccdULL;
  x ^= x >> 33;
  x *= 0xc4ceb9fe1a85ec53ULL;
  x ^= x >> 33;
  return x;
}

struct TableView {
  int64_t *keys;
  uint64_t *scores;
  uint32_t *index;
  uint16_t *next;
  uint16_t *head;
  int32_t *bucket_size;
  int32_t *refs;     // pin count of each row, or null when the table does not pin
  int64_t num_buckets;
  int64_t gid_base;  // buckets of the tables before this one: a bucket's global id
  int32_t cap;
  int32_t pad;
};
static_assert(sizeof(TableView) == 80, "the layout is built on the host");

__global__ void gid_kernel(const TableView *__restrict__ views,
                           const int64_t *__restrict__ keys,
                           const int64_t *__restrict__ tids, int64_t n,
                           int64_t *__restrict__ gid,
                           int64_t *__restrict__ local) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const TableView &v = views[tids[i]];
    const int64_t b =
        static_cast<int64_t>(hash64(keys[i]) % uint64_t(v.num_buckets));
    local[i] = b;
    gid[i] = v.gid_base + b;
  }
}

__global__ void bucket_ids_kernel(const int64_t *__restrict__ keys, int64_t n,
                                  int64_t num_buckets,
                                  int64_t *__restrict__ out) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    out[i] = static_cast<int64_t>(hash64(keys[i]) % uint64_t(num_buckets));
  }
}

// mode: 0 leave scores alone, 1 assign the given score, 2 add the given score
// (1 when none is given, which is how a frequency counts). With `pin`, a hit also
// takes a reference on its row.
__global__ void find_kernel(const TableView *__restrict__ views,
                            const int64_t *__restrict__ keys,
                            const int64_t *__restrict__ tids,
                            const uint64_t *__restrict__ in_scores, int mode,
                            int pin, int64_t n, int64_t *__restrict__ rows) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const TableView &v = views[tids[i]];
    const int64_t key = keys[i];
    const uint64_t h = hash64(key);
    const int64_t b = static_cast<int64_t>(h % uint64_t(v.num_buckets));
    const int c = static_cast<int>((h >> 32) % uint32_t(v.cap));
    const int64_t base = b * v.cap;
    int64_t row = -1;
    uint16_t cur = v.head[base + c];
    // The walk is bounded by the bucket size: a chain can hold no more nodes.
    for (int step = 0; cur != kNull && step < v.cap; ++step) {
      const int64_t s = base + cur;
      if (v.keys[s] == key) {
        row = static_cast<int64_t>(v.index[s]);
        if (mode == 1) {
          v.scores[s] = in_scores != nullptr ? in_scores[i] : 0;
        } else if (mode == 2) {
          v.scores[s] += in_scores != nullptr ? in_scores[i] : 1;
        }
        if (pin && v.refs != nullptr) atomicAdd(&v.refs[row], 1);
        break;
      }
      cur = v.next[s];
    }
    rows[i] = row;
  }
}

__global__ void pin_kernel(const TableView *__restrict__ views,
                           const int64_t *__restrict__ rows,
                           const int64_t *__restrict__ tids, int delta,
                           int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const TableView &v = views[tids[i]];
    if (v.refs != nullptr) atomicAdd(&v.refs[rows[i]], delta);
  }
}

__global__ void segment_need_kernel(const TableView *__restrict__ views,
                                    const int64_t *__restrict__ seg_table,
                                    const int64_t *__restrict__ seg_bucket,
                                    const int64_t *__restrict__ seg_count,
                                    int64_t u, int64_t *__restrict__ need) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < u;
       i += int64_t(gridDim.x) * blockDim.x) {
    const TableView &v = views[seg_table[i]];
    const int64_t free_slots = v.cap - v.bucket_size[seg_bucket[i]];
    const int64_t c = seg_count[i];
    need[i] = c > free_slots ? c - free_slots : 0;
  }
}

__device__ __forceinline__ void warp_min(uint64_t &sc, uint32_t &sl) {
  for (int off = 16; off > 0; off >>= 1) {
    const uint64_t osc = __shfl_xor_sync(0xffffffffu, sc, off);
    const uint32_t osl = __shfl_xor_sync(0xffffffffu, sl, off);
    if (osc < sc || (osc == sc && osl < sl)) {
      sc = osc;
      sl = osl;
    }
  }
}

// One warp per bucket segment. The segment is the contiguous run, in the sorted
// staged arrays, of the new keys that belong to one bucket of one table.
//
//   1. If the new keys outnumber the free slots, choose the victims first, all
//      of them, from the slots occupied before this step: lowest (score, slot)
//      first, skipping any the guard refuses or whose row is pinned. If there are
//      not enough, the segment is left untouched and flagged.
//   2. Then place: the first free slots take the first new keys, the victims'
//      slots take the rest. A victim's (key, row) is reported before its slot is
//      overwritten, and it is taken off its chain. With `pin_new` a placed key's
//      row starts with one reference.
__global__ void place_kernel(
    const TableView *__restrict__ views, const int64_t *__restrict__ skeys,
    const uint64_t *__restrict__ sscores, const int64_t *__restrict__ srows,
    const int64_t *__restrict__ seg_table,
    const int64_t *__restrict__ seg_bucket,
    const int64_t *__restrict__ seg_begin,
    const int64_t *__restrict__ seg_count,
    const int64_t *__restrict__ seg_voff, int64_t num_segments,
    int32_t *__restrict__ vict_slot, int64_t *__restrict__ vkeys,
    int64_t *__restrict__ vrows, int32_t *__restrict__ seg_fail,
    embedding_cache::EvictionGuardView guard, int pin_new) {
  __shared__ uint8_t taken[kWarpsPerBlock][kMaxBucketCapacity];

  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int64_t warp_id = int64_t(blockIdx.x) * kWarpsPerBlock + w;
  const int64_t total_warps = int64_t(gridDim.x) * kWarpsPerBlock;

  for (int64_t seg = warp_id; seg < num_segments; seg += total_warps) {
    const int64_t t = seg_table[seg];
    const TableView &v = views[t];
    const int64_t b = seg_bucket[seg];
    const int64_t begin = seg_begin[seg];
    const int64_t c = seg_count[seg];
    const int64_t voff = seg_voff[seg];
    const int64_t base = b * v.cap;
    const int32_t s0 = v.bucket_size[b];
    const int64_t f = v.cap - s0;
    const int64_t k = c > f ? c - f : 0;

    // 1. victims
    bool fail = false;
    if (k > 0) {
      for (int j = lane; j < v.cap; j += 32) taken[w][j] = 0;
      __syncwarp();
      int64_t sel = 0;
      while (sel < k) {
        uint64_t bs = ~0ULL;
        uint32_t bl = 0xFFFFFFFFu;
        for (int j = lane; j < s0; j += 32) {
          if (!taken[w][j]) {
            const uint64_t sc = v.scores[base + j];
            if (sc < bs || (sc == bs && uint32_t(j) < bl)) {
              bs = sc;
              bl = uint32_t(j);
            }
          }
        }
        warp_min(bs, bl);
        if (bl == 0xFFFFFFFFu) {
          fail = true;
          break;
        }
        bool refused = false;
        if (lane == 0) {
          if (v.refs != nullptr && v.refs[v.index[base + bl]] > 0) {
            refused = true;
          } else if (guard.tables != nullptr) {
            refused = guard.refuses(v.keys[base + bl], t);
          }
        }
        refused = __shfl_sync(0xffffffffu, refused, 0);
        if (lane == 0) taken[w][bl] = 1;
        __syncwarp();
        if (!refused) {
          if (lane == 0) vict_slot[voff + sel] = int32_t(bl);
          ++sel;
        }
      }
    }
    if (fail) {
      if (lane == 0) seg_fail[seg] = 1;
      __syncwarp();
      continue;
    }

    // 2. place
    if (lane == 0) {
      int32_t s = s0;
      int64_t vi = 0;
      for (int64_t j = 0; j < c; ++j) {
        const int64_t key = skeys[begin + j];
        int64_t slot;
        if (s < v.cap) {
          slot = s++;
        } else {
          slot = vict_slot[voff + vi];
          const int64_t sl = base + slot;
          const int64_t vkey = v.keys[sl];
          vkeys[voff + vi] = vkey;
          vrows[voff + vi] = static_cast<int64_t>(v.index[sl]);
          ++vi;
          // off its chain
          const int chv = static_cast<int>((hash64(vkey) >> 32) % uint32_t(v.cap));
          uint16_t cur = v.head[base + chv];
          if (cur == uint16_t(slot)) {
            v.head[base + chv] = v.next[sl];
          } else {
            while (cur != kNull) {
              const uint16_t nx = v.next[base + cur];
              if (nx == uint16_t(slot)) {
                v.next[base + cur] = v.next[sl];
                break;
              }
              cur = nx;
            }
          }
        }
        const int64_t sl = base + slot;
        const int64_t row = srows[begin + j];
        v.keys[sl] = key;
        v.scores[sl] = sscores[begin + j];
        v.index[sl] = static_cast<uint32_t>(row);
        const int chn = static_cast<int>((hash64(key) >> 32) % uint32_t(v.cap));
        v.next[sl] = v.head[base + chn];
        v.head[base + chn] = uint16_t(slot);
        if (pin_new && v.refs != nullptr) v.refs[row] = 1;
      }
      v.bucket_size[b] = s;
    }
    __syncwarp();
  }
}

// ---------------------------------------------------------------------------
// torch glue
// ---------------------------------------------------------------------------

inline void check_cuda(const at::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

inline int grid_for(int64_t n, int block) {
  const int64_t blocks = (n + block - 1) / block;
  return static_cast<int>(std::min<int64_t>(std::max<int64_t>(blocks, 1), 65535));
}

inline const TableView *views_ptr(const at::Tensor &views) {
  check_cuda(views, "views");
  TORCH_CHECK(views.scalar_type() == at::kByte &&
                  views.numel() % sizeof(TableView) == 0,
              "views must be a byte tensor built by chained_make_views");
  return reinterpret_cast<const TableView *>(views.data_ptr());
}

// One TableView per table, built on the host and copied over. Call again, and keep
// the new tensor, whenever a table's arrays are replaced.
at::Tensor chained_make_views(std::vector<at::Tensor> keys,
                              std::vector<at::Tensor> scores,
                              std::vector<at::Tensor> index,
                              std::vector<at::Tensor> next,
                              std::vector<at::Tensor> head,
                              std::vector<at::Tensor> bucket_size,
                              std::vector<c10::optional<at::Tensor>> refs,
                              std::vector<int64_t> num_buckets, int64_t cap) {
  const size_t nt = keys.size();
  TORCH_CHECK(scores.size() == nt && index.size() == nt && next.size() == nt &&
                  head.size() == nt && bucket_size.size() == nt &&
                  refs.size() == nt && num_buckets.size() == nt,
              "one entry per table in every list");
  TORCH_CHECK(cap > 0 && cap <= kMaxBucketCapacity,
              "bucket capacity must be in [1, ", kMaxBucketCapacity, "]");
  std::vector<TableView> host(nt);
  int64_t gid_base = 0;
  for (size_t t = 0; t < nt; ++t) {
    check_cuda(keys[t], "table keys");
    check_cuda(scores[t], "table scores");
    check_cuda(index[t], "table index");
    check_cuda(next[t], "table next");
    check_cuda(head[t], "table head");
    check_cuda(bucket_size[t], "table bucket_size");
    TORCH_CHECK(keys[t].element_size() == 8 && scores[t].element_size() == 8,
                "keys and scores must be 8 bytes wide");
    TORCH_CHECK(index[t].element_size() == 4, "index must be 4 bytes wide");
    TORCH_CHECK(next[t].element_size() == 2 && head[t].element_size() == 2,
                "next and head must be 2 bytes wide");
    TORCH_CHECK(bucket_size[t].scalar_type() == at::kInt,
                "bucket_size must be int32");
    const int64_t slots = num_buckets[t] * cap;
    TORCH_CHECK(keys[t].numel() == slots && scores[t].numel() == slots &&
                    index[t].numel() == slots && next[t].numel() == slots &&
                    head[t].numel() == slots &&
                    bucket_size[t].numel() == num_buckets[t],
                "table arrays do not match num_buckets * bucket_capacity");
    TableView &v = host[t];
    v.keys = reinterpret_cast<int64_t *>(keys[t].data_ptr());
    v.scores = reinterpret_cast<uint64_t *>(scores[t].data_ptr());
    v.index = reinterpret_cast<uint32_t *>(index[t].data_ptr());
    v.next = reinterpret_cast<uint16_t *>(next[t].data_ptr());
    v.head = reinterpret_cast<uint16_t *>(head[t].data_ptr());
    v.bucket_size = bucket_size[t].data_ptr<int32_t>();
    if (refs[t].has_value() && refs[t]->defined()) {
      check_cuda(*refs[t], "table refs");
      TORCH_CHECK(refs[t]->scalar_type() == at::kInt, "refs must be int32");
      v.refs = refs[t]->data_ptr<int32_t>();
    } else {
      v.refs = nullptr;
    }
    v.num_buckets = num_buckets[t];
    v.gid_base = gid_base;
    v.cap = static_cast<int32_t>(cap);
    v.pad = 0;
    gid_base += num_buckets[t];
  }
  auto out = at::empty({int64_t(nt * sizeof(TableView))},
                       at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  if (nt > 0) {
    C10_CUDA_CHECK(cudaMemcpyAsync(out.data_ptr(), host.data(),
                                   nt * sizeof(TableView),
                                   cudaMemcpyHostToDevice,
                                   at::cuda::getCurrentCUDAStream()));
    // The host vector goes out of scope: wait for the copy.
    C10_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
  }
  return out;
}

at::Tensor chained_bucket_ids(at::Tensor keys, int64_t num_buckets) {
  check_cuda(keys, "keys");
  TORCH_CHECK(keys.element_size() == 8, "keys must be 8 bytes wide");
  auto out = at::empty({keys.numel()}, keys.options().dtype(at::kLong));
  const int64_t n = keys.numel();
  if (n == 0) return out;
  auto stream = at::cuda::getCurrentCUDAStream();
  bucket_ids_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      reinterpret_cast<const int64_t *>(keys.data_ptr()), n, num_buckets,
      out.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

// (gid, local bucket) of each key: the bucket's id among all tables' buckets, and
// within its own table.
std::tuple<at::Tensor, at::Tensor> chained_gids(at::Tensor views,
                                                at::Tensor keys,
                                                at::Tensor tids) {
  check_cuda(keys, "keys");
  check_cuda(tids, "table ids");
  TORCH_CHECK(keys.element_size() == 8 && tids.scalar_type() == at::kLong &&
                  keys.numel() == tids.numel(),
              "keys and table ids must be 8 bytes wide and as many");
  const int64_t n = keys.numel();
  auto gid = at::empty({n}, keys.options().dtype(at::kLong));
  auto local = at::empty({n}, keys.options().dtype(at::kLong));
  if (n == 0) return {gid, local};
  auto stream = at::cuda::getCurrentCUDAStream();
  gid_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      views_ptr(views), reinterpret_cast<const int64_t *>(keys.data_ptr()),
      tids.data_ptr<int64_t>(), n, gid.data_ptr<int64_t>(),
      local.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {gid, local};
}

at::Tensor chained_find(at::Tensor views, at::Tensor keys, at::Tensor tids,
                        c10::optional<at::Tensor> scores, int64_t mode,
                        bool pin) {
  check_cuda(keys, "keys");
  check_cuda(tids, "table ids");
  TORCH_CHECK(keys.element_size() == 8, "keys must be 8 bytes wide");
  TORCH_CHECK(tids.scalar_type() == at::kLong && tids.numel() == keys.numel(),
              "table ids must be int64, one per key");
  TORCH_CHECK(mode >= 0 && mode <= 2, "mode must be 0, 1 or 2");
  const uint64_t *sc = nullptr;
  if (scores.has_value() && scores->defined()) {
    check_cuda(*scores, "scores");
    TORCH_CHECK(scores->numel() == keys.numel() && scores->element_size() == 8,
                "scores must have one 8-byte entry per key");
    sc = reinterpret_cast<const uint64_t *>(scores->data_ptr());
  }
  const int64_t n = keys.numel();
  auto rows = at::empty({n}, keys.options().dtype(at::kLong));
  if (n == 0) return rows;
  auto stream = at::cuda::getCurrentCUDAStream();
  find_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      views_ptr(views), reinterpret_cast<const int64_t *>(keys.data_ptr()),
      tids.data_ptr<int64_t>(), sc, static_cast<int>(mode), pin ? 1 : 0, n,
      rows.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rows;
}

// Add `delta` to the pin count of each (table, row).
void chained_pin(at::Tensor views, at::Tensor rows, at::Tensor tids,
                 int64_t delta) {
  check_cuda(rows, "rows");
  check_cuda(tids, "table ids");
  TORCH_CHECK(rows.scalar_type() == at::kLong && tids.scalar_type() == at::kLong &&
                  rows.numel() == tids.numel(),
              "rows and table ids must be int64 and as many");
  const int64_t n = rows.numel();
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  pin_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      views_ptr(views), rows.data_ptr<int64_t>(), tids.data_ptr<int64_t>(),
      static_cast<int>(delta), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor chained_segment_need(at::Tensor views, at::Tensor seg_table,
                                at::Tensor seg_bucket, at::Tensor seg_count) {
  check_cuda(seg_table, "seg_table");
  check_cuda(seg_bucket, "seg_bucket");
  check_cuda(seg_count, "seg_count");
  const int64_t u = seg_table.numel();
  auto need = at::empty({u}, seg_table.options());
  if (u == 0) return need;
  auto stream = at::cuda::getCurrentCUDAStream();
  segment_need_kernel<<<grid_for(u, 256), 256, 0, stream>>>(
      views_ptr(views), seg_table.data_ptr<int64_t>(),
      seg_bucket.data_ptr<int64_t>(), seg_count.data_ptr<int64_t>(), u,
      need.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return need;
}

void chained_place(at::Tensor views, at::Tensor skeys, at::Tensor sscores,
                   at::Tensor srows, at::Tensor seg_table,
                   at::Tensor seg_bucket, at::Tensor seg_begin,
                   at::Tensor seg_count, at::Tensor seg_voff,
                   at::Tensor vict_slot, at::Tensor vkeys, at::Tensor vrows,
                   at::Tensor seg_fail, int64_t guard_tables,
                   int64_t guard_num_tables, bool pin_new) {
  for (auto *t : {&skeys, &sscores, &srows, &seg_table, &seg_bucket, &seg_begin,
                  &seg_count, &seg_voff, &vict_slot, &vkeys, &vrows,
                  &seg_fail}) {
    check_cuda(*t, "commit argument");
  }
  const int64_t u = seg_bucket.numel();
  if (u == 0) return;
  embedding_cache::EvictionGuardView guard{
      reinterpret_cast<const void *>(guard_tables),
      static_cast<int32_t>(guard_num_tables)};
  auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = grid_for(u, kWarpsPerBlock);
  place_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
      views_ptr(views), reinterpret_cast<const int64_t *>(skeys.data_ptr()),
      reinterpret_cast<const uint64_t *>(sscores.data_ptr()),
      srows.data_ptr<int64_t>(), seg_table.data_ptr<int64_t>(),
      seg_bucket.data_ptr<int64_t>(), seg_begin.data_ptr<int64_t>(),
      seg_count.data_ptr<int64_t>(), seg_voff.data_ptr<int64_t>(), u,
      vict_slot.data_ptr<int32_t>(), vkeys.data_ptr<int64_t>(),
      vrows.data_ptr<int64_t>(), seg_fail.data_ptr<int32_t>(), guard,
      pin_new ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace chained
} // namespace dyn_emb

void bind_chained_table(py::module &m) {
  m.def("chained_make_views", &dyn_emb::chained::chained_make_views,
        "build the device array of table views", py::arg("keys"),
        py::arg("scores"), py::arg("index"), py::arg("next"), py::arg("head"),
        py::arg("bucket_size"), py::arg("refs"), py::arg("num_buckets"),
        py::arg("bucket_capacity"));
  m.def("chained_bucket_ids", &dyn_emb::chained::chained_bucket_ids,
        "bucket of each key: hash(key) % num_buckets", py::arg("keys"),
        py::arg("num_buckets"));
  m.def("chained_gids", &dyn_emb::chained::chained_gids,
        "(global bucket id, bucket within its table) of each (key, table)",
        py::arg("views"), py::arg("keys"), py::arg("table_ids"));
  m.def("chained_find", &dyn_emb::chained::chained_find,
        "row of each key in its chained table (-1 if absent); mode 0 leaves "
        "scores alone, 1 assigns the given score to hits, 2 adds it; pin takes "
        "a reference on the rows found",
        py::arg("views"), py::arg("keys"), py::arg("table_ids"),
        py::arg("scores") = py::none(), py::arg("mode") = 0,
        py::arg("pin") = false);
  m.def("chained_pin", &dyn_emb::chained::chained_pin,
        "add delta to the pin count of each (table, row)", py::arg("views"),
        py::arg("rows"), py::arg("table_ids"), py::arg("delta"));
  m.def("chained_segment_need", &dyn_emb::chained::chained_segment_need,
        "how many members each bucket segment has to evict", py::arg("views"),
        py::arg("seg_table"), py::arg("seg_bucket"), py::arg("seg_count"));
  m.def("chained_place", &dyn_emb::chained::chained_place,
        "place the sorted staged keys, one warp per bucket segment, evicting "
        "the lowest-scored members of a bucket that has no room",
        py::arg("views"), py::arg("skeys"), py::arg("sscores"),
        py::arg("srows"), py::arg("seg_table"), py::arg("seg_bucket"),
        py::arg("seg_begin"), py::arg("seg_count"), py::arg("seg_voff"),
        py::arg("vict_slot"), py::arg("vkeys"), py::arg("vrows"),
        py::arg("seg_fail"), py::arg("guard_tables") = 0,
        py::arg("guard_num_tables") = 0, py::arg("pin_new") = false);
}
