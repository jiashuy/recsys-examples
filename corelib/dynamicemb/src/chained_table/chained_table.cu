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

struct View {
  int64_t *keys;
  uint64_t *scores;
  uint32_t *index;
  uint16_t *next;
  uint16_t *head;
  int32_t *bucket_size;
  int64_t num_buckets;
  int32_t cap;
};

__global__ void bucket_ids_kernel(const int64_t *__restrict__ keys, int64_t n,
                                  int64_t num_buckets,
                                  int64_t *__restrict__ out) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    out[i] = static_cast<int64_t>(hash64(keys[i]) % uint64_t(num_buckets));
  }
}

// mode: 0 leave scores alone, 1 assign the given score, 2 add the given score
// (1 when none is given, which is how a frequency counts).
__global__ void find_kernel(View v, const int64_t *__restrict__ keys,
                            const uint64_t *__restrict__ in_scores, int mode,
                            int64_t n, int64_t *__restrict__ rows) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
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
        break;
      }
      cur = v.next[s];
    }
    rows[i] = row;
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
// staged arrays, of the new keys that belong to one bucket.
//
//   1. If the new keys outnumber the free slots, choose the victims first, all
//      of them, from the slots occupied before this step: lowest (score, slot)
//      first, skipping any the guard refuses. If there are not enough, the
//      segment is left untouched and flagged.
//   2. Then place: the first free slots take the first new keys, the victims'
//      slots take the rest. A victim's (key, row) is reported before its slot is
//      overwritten, and it is taken off its chain.
__global__ void place_kernel(
    View v, const int64_t *__restrict__ skeys,
    const uint64_t *__restrict__ sscores, const int64_t *__restrict__ srows,
    const int64_t *__restrict__ seg_bucket,
    const int64_t *__restrict__ seg_begin,
    const int64_t *__restrict__ seg_count,
    const int64_t *__restrict__ seg_voff, int64_t num_segments,
    int32_t *__restrict__ vict_slot, int64_t *__restrict__ vkeys,
    int64_t *__restrict__ vrows, int32_t *__restrict__ seg_fail,
    embedding_cache::EvictionGuardView guard, int64_t guard_table) {
  __shared__ uint8_t taken[kWarpsPerBlock][kMaxBucketCapacity];

  const int lane = threadIdx.x & 31;
  const int w = threadIdx.x >> 5;
  const int64_t warp_id = int64_t(blockIdx.x) * kWarpsPerBlock + w;
  const int64_t total_warps = int64_t(gridDim.x) * kWarpsPerBlock;

  for (int64_t seg = warp_id; seg < num_segments; seg += total_warps) {
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
        if (guard.tables != nullptr) {
          if (lane == 0) refused = guard.refuses(v.keys[base + bl], guard_table);
          refused = __shfl_sync(0xffffffffu, refused, 0);
        }
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
        v.keys[sl] = key;
        v.scores[sl] = sscores[begin + j];
        v.index[sl] = static_cast<uint32_t>(srows[begin + j]);
        const int chn = static_cast<int>((hash64(key) >> 32) % uint32_t(v.cap));
        v.next[sl] = v.head[base + chn];
        v.head[base + chn] = uint16_t(slot);
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

inline View make_view(at::Tensor &tkeys, at::Tensor &tscores,
                      at::Tensor &tindex, at::Tensor &tnext, at::Tensor &thead,
                      at::Tensor &tbsize, int64_t num_buckets, int64_t cap) {
  check_cuda(tkeys, "table keys");
  check_cuda(tscores, "table scores");
  check_cuda(tindex, "table index");
  check_cuda(tnext, "table next");
  check_cuda(thead, "table head");
  check_cuda(tbsize, "table bucket_size");
  TORCH_CHECK(tkeys.element_size() == 8 && tscores.element_size() == 8,
              "keys and scores must be 8 bytes wide");
  TORCH_CHECK(tindex.element_size() == 4, "index must be 4 bytes wide");
  TORCH_CHECK(tnext.element_size() == 2 && thead.element_size() == 2,
              "next and head must be 2 bytes wide");
  TORCH_CHECK(tbsize.scalar_type() == at::kInt, "bucket_size must be int32");
  TORCH_CHECK(cap > 0 && cap <= kMaxBucketCapacity,
              "bucket capacity must be in [1, ", kMaxBucketCapacity, "]");
  const int64_t slots = num_buckets * cap;
  TORCH_CHECK(tkeys.numel() == slots && tscores.numel() == slots &&
                  tindex.numel() == slots && tnext.numel() == slots &&
                  thead.numel() == slots && tbsize.numel() == num_buckets,
              "table arrays do not match num_buckets * bucket_capacity");
  View v;
  v.keys = reinterpret_cast<int64_t *>(tkeys.data_ptr());
  v.scores = reinterpret_cast<uint64_t *>(tscores.data_ptr());
  v.index = reinterpret_cast<uint32_t *>(tindex.data_ptr());
  v.next = reinterpret_cast<uint16_t *>(tnext.data_ptr());
  v.head = reinterpret_cast<uint16_t *>(thead.data_ptr());
  v.bucket_size = tbsize.data_ptr<int32_t>();
  v.num_buckets = num_buckets;
  v.cap = static_cast<int32_t>(cap);
  return v;
}

inline int grid_for(int64_t n, int block) {
  const int64_t blocks = (n + block - 1) / block;
  return static_cast<int>(std::min<int64_t>(std::max<int64_t>(blocks, 1), 65535));
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

at::Tensor chained_find(at::Tensor keys, at::Tensor tkeys, at::Tensor tscores,
                        at::Tensor tindex, at::Tensor tnext, at::Tensor thead,
                        at::Tensor tbsize, int64_t num_buckets, int64_t cap,
                        c10::optional<at::Tensor> scores, int64_t mode) {
  check_cuda(keys, "keys");
  TORCH_CHECK(keys.element_size() == 8, "keys must be 8 bytes wide");
  TORCH_CHECK(mode >= 0 && mode <= 2, "mode must be 0, 1 or 2");
  View v = make_view(tkeys, tscores, tindex, tnext, thead, tbsize, num_buckets,
                     cap);
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
      v, reinterpret_cast<const int64_t *>(keys.data_ptr()), sc,
      static_cast<int>(mode), n, rows.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return rows;
}

void chained_place(at::Tensor tkeys, at::Tensor tscores, at::Tensor tindex,
                   at::Tensor tnext, at::Tensor thead, at::Tensor tbsize,
                   int64_t num_buckets, int64_t cap, at::Tensor skeys,
                   at::Tensor sscores, at::Tensor srows, at::Tensor seg_bucket,
                   at::Tensor seg_begin, at::Tensor seg_count,
                   at::Tensor seg_voff, at::Tensor vict_slot, at::Tensor vkeys,
                   at::Tensor vrows, at::Tensor seg_fail, int64_t guard_tables,
                   int64_t guard_num_tables, int64_t guard_table) {
  View v = make_view(tkeys, tscores, tindex, tnext, thead, tbsize, num_buckets,
                     cap);
  for (auto *t : {&skeys, &sscores, &srows, &seg_bucket, &seg_begin, &seg_count,
                  &seg_voff, &vict_slot, &vkeys, &vrows, &seg_fail}) {
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
      v, reinterpret_cast<const int64_t *>(skeys.data_ptr()),
      reinterpret_cast<const uint64_t *>(sscores.data_ptr()),
      srows.data_ptr<int64_t>(), seg_bucket.data_ptr<int64_t>(),
      seg_begin.data_ptr<int64_t>(), seg_count.data_ptr<int64_t>(),
      seg_voff.data_ptr<int64_t>(), u, vict_slot.data_ptr<int32_t>(),
      vkeys.data_ptr<int64_t>(), vrows.data_ptr<int64_t>(),
      seg_fail.data_ptr<int32_t>(), guard, guard_table);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace chained
} // namespace dyn_emb

void bind_chained_table(py::module &m) {
  m.def("chained_bucket_ids", &dyn_emb::chained::chained_bucket_ids,
        "bucket of each key: hash(key) % num_buckets", py::arg("keys"),
        py::arg("num_buckets"));
  m.def("chained_find", &dyn_emb::chained::chained_find,
        "row of each key in a chained bucket table (-1 if absent); mode 0 "
        "leaves scores alone, 1 assigns the given score to hits, 2 adds it",
        py::arg("keys"), py::arg("table_keys"), py::arg("table_scores"),
        py::arg("table_index"), py::arg("table_next"), py::arg("table_head"),
        py::arg("bucket_size"), py::arg("num_buckets"),
        py::arg("bucket_capacity"), py::arg("scores") = py::none(),
        py::arg("mode") = 0);
  m.def("chained_place", &dyn_emb::chained::chained_place,
        "place the sorted staged keys, one warp per bucket segment, evicting "
        "the lowest-scored members of a bucket that has no room",
        py::arg("table_keys"), py::arg("table_scores"), py::arg("table_index"),
        py::arg("table_next"), py::arg("table_head"), py::arg("bucket_size"),
        py::arg("num_buckets"), py::arg("bucket_capacity"), py::arg("skeys"),
        py::arg("sscores"), py::arg("srows"), py::arg("seg_bucket"),
        py::arg("seg_begin"), py::arg("seg_count"), py::arg("seg_voff"),
        py::arg("vict_slot"), py::arg("vkeys"), py::arg("vrows"),
        py::arg("seg_fail"), py::arg("guard_tables") = 0,
        py::arg("guard_num_tables") = 0, py::arg("guard_table") = 0);
}
