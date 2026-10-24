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

#include <cub/cub.cuh>

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
  uint32_t *free_rows;  // the stack of free rows ...
  int32_t *top;         // ... and how many are on it (on the device)
  int64_t num_buckets;
  int64_t gid_base;  // buckets of the tables before this one: a bucket's global id
  int32_t cap;
  int32_t pad;
};
static_assert(sizeof(TableView) == 96, "the layout is built on the host");

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
    const int64_t *__restrict__ seg_voff, const int64_t *__restrict__ nseg_ptr,
    const int32_t *__restrict__ abort, int32_t *__restrict__ vict_slot,
    int64_t *__restrict__ vkeys, int64_t *__restrict__ vrows,
    int64_t *__restrict__ vtabs, int32_t *__restrict__ seg_fail,
    unsigned long long *__restrict__ nfail,
    embedding_cache::EvictionGuardView guard, int pin_new) {
  __shared__ uint8_t taken[kWarpsPerBlock][kMaxBucketCapacity];

  if (*abort != 0) return;
  const int64_t num_segments = *nseg_ptr;
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
      if (lane == 0) {
        seg_fail[seg] = 1;
        atomicAdd(nfail, 1ULL);
      }
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
          vtabs[voff + vi] = t;
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

// -- the rest of a step's bookkeeping, as kernels ------------------------------

// positions of the misses, compacted in input order
__global__ void miss_flag_kernel(const int64_t *__restrict__ rows, int64_t n,
                                 int32_t *__restrict__ flag) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    flag[i] = rows[i] < 0 ? 1 : 0;
  }
}

__global__ void stage_scatter_kernel(
    const int32_t *__restrict__ flag, const int32_t *__restrict__ pos,
    const int64_t *__restrict__ keys, const int64_t *__restrict__ tids,
    const int64_t *__restrict__ scores, int64_t fill, int64_t n,
    int64_t *__restrict__ okeys, int64_t *__restrict__ otids,
    int64_t *__restrict__ oscores, int64_t *__restrict__ opos,
    int64_t *__restrict__ total) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    if (flag[i]) {
      const int64_t o = pos[i];
      okeys[o] = keys[i];
      otids[o] = tids[i];
      oscores[o] = scores != nullptr ? scores[i] : fill;
      opos[o] = i;
    }
    if (i == n - 1) total[0] = int64_t(pos[i]) + flag[i];
  }
}

__global__ void gather_kernel(const int64_t *__restrict__ perm,
                              const int64_t *__restrict__ keys,
                              const int64_t *__restrict__ tids,
                              const int64_t *__restrict__ scores,
                              const int64_t *__restrict__ local, int64_t n,
                              int64_t *__restrict__ skeys,
                              int64_t *__restrict__ stids,
                              int64_t *__restrict__ sscores,
                              int64_t *__restrict__ slocal) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t j = perm[i];
    skeys[i] = keys[j];
    stids[i] = tids[j];
    sscores[i] = scores[j];
    slocal[i] = local[j];
  }
}

// seg_table, seg_bucket and need of each run of equal bucket ids; 0 past the last
__global__ void segment_info_kernel(
    const TableView *__restrict__ views, const int64_t *__restrict__ stids,
    const int64_t *__restrict__ slocal, const int64_t *__restrict__ seg_begin,
    const int64_t *__restrict__ counts, const int64_t *__restrict__ nruns_ptr,
    int64_t n, int64_t *__restrict__ seg_table, int64_t *__restrict__ seg_bucket,
    int64_t *__restrict__ need) {
  const int64_t nruns = *nruns_ptr;
  for (int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; r < n;
       r += int64_t(gridDim.x) * blockDim.x) {
    if (r < nruns) {
      const int64_t b = seg_begin[r];
      const int64_t t = stids[b];
      const int64_t bucket = slocal[b];
      const TableView &v = views[t];
      const int64_t free_slots = v.cap - v.bucket_size[bucket];
      const int64_t c = counts[r];
      seg_table[r] = t;
      seg_bucket[r] = bucket;
      need[r] = c > free_slots ? c - free_slots : 0;
    } else {
      seg_table[r] = 0;
      seg_bucket[r] = 0;
      need[r] = 0;
    }
  }
}

// where each table's keys start in the sorted staged arrays (and n at the end)
__global__ void table_start_kernel(const TableView *__restrict__ views,
                                   const int64_t *__restrict__ sgid, int64_t n,
                                   int64_t num_tables,
                                   int64_t *__restrict__ tstart) {
  const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t > num_tables) return;
  if (t == num_tables) {
    tstart[t] = n;
    return;
  }
  const int64_t target = views[t].gid_base;
  int64_t lo = 0, hi = n;
  while (lo < hi) {
    const int64_t mid = (lo + hi) / 2;
    if (sgid[mid] < target) lo = mid + 1; else hi = mid;
  }
  tstart[t] = lo;
}

__global__ void check_pool_kernel(const TableView *__restrict__ views,
                                  const int64_t *__restrict__ tstart,
                                  int64_t num_tables, int32_t *__restrict__ abort) {
  const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= num_tables) return;
  const int64_t c = tstart[t + 1] - tstart[t];
  if (c > int64_t(*views[t].top)) atomicExch(abort, 1);
}

__global__ void assign_rows_kernel(const TableView *__restrict__ views,
                                   const int64_t *__restrict__ stids,
                                   const int64_t *__restrict__ tstart,
                                   const int32_t *__restrict__ abort, int64_t n,
                                   int64_t *__restrict__ srows) {
  if (*abort != 0) return;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const int64_t t = stids[i];
    const TableView &v = views[t];
    const int64_t c = tstart[t + 1] - tstart[t];
    srows[i] = static_cast<int64_t>(v.free_rows[int64_t(*v.top) - c + (i - tstart[t])]);
  }
}

__global__ void pop_rows_kernel(const TableView *__restrict__ views,
                                const int64_t *__restrict__ tstart,
                                int64_t num_tables,
                                const int32_t *__restrict__ abort) {
  const int64_t t = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= num_tables || *abort != 0) return;
  *views[t].top -= int32_t(tstart[t + 1] - tstart[t]);
}

__global__ void unpermute_kernel(const int64_t *__restrict__ perm,
                                 const int64_t *__restrict__ srows, int64_t n,
                                 int64_t *__restrict__ rows) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    rows[perm[i]] = srows[i];
  }
}

__global__ void stats_kernel(const int64_t *__restrict__ voff,
                             const int64_t *__restrict__ need,
                             const int64_t *__restrict__ nruns_ptr, int64_t n,
                             const int32_t *__restrict__ abort,
                             int64_t *__restrict__ stats) {
  stats[0] = voff[n - 1] + need[n - 1];
  stats[1] = *nruns_ptr;
  stats[3] = *abort;
}

__global__ void release_kernel(const TableView *__restrict__ views, int64_t table,
                               const int64_t *__restrict__ rows, int64_t n) {
  const TableView &v = views[table];
  const int64_t top = *v.top;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    v.free_rows[top + i] = static_cast<uint32_t>(rows[i]);
  }
}

__global__ void push_rows_kernel(const TableView *__restrict__ views,
                                 int64_t table, int64_t n) {
  *views[table].top += int32_t(n);
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
                              std::vector<at::Tensor> free_rows,
                              at::Tensor tops,
                              std::vector<int64_t> num_buckets, int64_t cap) {
  const size_t nt = keys.size();
  TORCH_CHECK(scores.size() == nt && index.size() == nt && next.size() == nt &&
                  head.size() == nt && bucket_size.size() == nt &&
                  refs.size() == nt && free_rows.size() == nt &&
                  num_buckets.size() == nt,
              "one entry per table in every list");
  TORCH_CHECK(cap > 0 && cap <= kMaxBucketCapacity,
              "bucket capacity must be in [1, ", kMaxBucketCapacity, "]");
  check_cuda(tops, "tops");
  TORCH_CHECK(tops.scalar_type() == at::kInt && tops.numel() == int64_t(nt),
              "tops must be one int32 per table");
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
    check_cuda(free_rows[t], "free rows");
    TORCH_CHECK(free_rows[t].element_size() == 4, "free rows must be 4 bytes wide");
    v.free_rows = reinterpret_cast<uint32_t *>(free_rows[t].data_ptr());
    v.top = tops.data_ptr<int32_t>() + t;
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

inline at::Tensor temp_storage(size_t bytes) {
  return at::empty({int64_t(std::max<size_t>(bytes, 1))},
                   at::TensorOptions().dtype(at::kByte).device(at::kCUDA));
}

inline at::Tensor device_scalar(int64_t v, const at::Tensor &like) {
  return at::full({1}, v, like.options().dtype(at::kLong));
}

// Place with a segment count known on the host: for rebuilding a table.
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
  auto nseg = device_scalar(u, seg_table);
  auto abort = at::zeros({1}, seg_table.options().dtype(at::kInt));
  auto vtabs = at::empty({std::max<int64_t>(vkeys.numel(), 1)}, seg_table.options());
  auto nfail = at::zeros({1}, seg_table.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  place_kernel<<<grid_for(u, kWarpsPerBlock), kWarpsPerBlock * 32, 0, stream>>>(
      views_ptr(views), reinterpret_cast<const int64_t *>(skeys.data_ptr()),
      reinterpret_cast<const uint64_t *>(sscores.data_ptr()),
      srows.data_ptr<int64_t>(), seg_table.data_ptr<int64_t>(),
      seg_bucket.data_ptr<int64_t>(), seg_begin.data_ptr<int64_t>(),
      seg_count.data_ptr<int64_t>(), seg_voff.data_ptr<int64_t>(),
      nseg.data_ptr<int64_t>(), abort.data_ptr<int32_t>(),
      vict_slot.data_ptr<int32_t>(), vkeys.data_ptr<int64_t>(),
      vrows.data_ptr<int64_t>(), vtabs.data_ptr<int64_t>(),
      seg_fail.data_ptr<int32_t>(),
      reinterpret_cast<unsigned long long *>(nfail.data_ptr<int64_t>()), guard,
      pin_new ? 1 : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// After find: compact the misses, in input order. Returns the number of misses and
// the staged (keys, table ids, scores, positions), each sized to the input; only
// the first `m` entries mean anything. One host synchronisation, for `m`.
std::tuple<int64_t, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
chained_stage(at::Tensor rows, at::Tensor keys, at::Tensor tids,
              c10::optional<at::Tensor> scores, int64_t fill) {
  check_cuda(rows, "rows");
  check_cuda(keys, "keys");
  check_cuda(tids, "table ids");
  const int64_t n = keys.numel();
  auto okeys = at::empty({n}, keys.options().dtype(at::kLong));
  auto otids = at::empty({n}, keys.options().dtype(at::kLong));
  auto oscores = at::empty({n}, keys.options().dtype(at::kLong));
  auto opos = at::empty({n}, keys.options().dtype(at::kLong));
  if (n == 0) return {0, okeys, otids, oscores, opos};
  const int64_t *sc = nullptr;
  if (scores.has_value() && scores->defined()) {
    check_cuda(*scores, "scores");
    sc = reinterpret_cast<const int64_t *>(scores->data_ptr());
  }
  auto flag = at::empty({n}, keys.options().dtype(at::kInt));
  auto pos = at::empty({n}, keys.options().dtype(at::kInt));
  auto total = at::zeros({1}, keys.options().dtype(at::kLong));
  auto stream = at::cuda::getCurrentCUDAStream();
  miss_flag_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      rows.data_ptr<int64_t>(), n, flag.data_ptr<int32_t>());
  size_t tb = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, tb, flag.data_ptr<int32_t>(),
                                pos.data_ptr<int32_t>(), int(n), stream);
  auto tmp = temp_storage(tb);
  cub::DeviceScan::ExclusiveSum(tmp.data_ptr(), tb, flag.data_ptr<int32_t>(),
                                pos.data_ptr<int32_t>(), int(n), stream);
  stage_scatter_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      flag.data_ptr<int32_t>(), pos.data_ptr<int32_t>(),
      reinterpret_cast<const int64_t *>(keys.data_ptr()),
      tids.data_ptr<int64_t>(), sc, fill, n, okeys.data_ptr<int64_t>(),
      otids.data_ptr<int64_t>(), oscores.data_ptr<int64_t>(),
      opos.data_ptr<int64_t>(), total.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto host = total.to(at::kCPU);  // the one synchronisation
  return {host.item<int64_t>(), okeys, otids, oscores, opos};
}

// Sort the staged keys by bucket and cut them into one run per bucket. Everything
// returned is sized to the input and stays on the device; `seg_*`, `need`, `voff`
// are meaningful for the first `num_runs` entries (0 after).
//
//   perm, sgid, skeys, stids, sscores, seg_table, seg_bucket, seg_begin, counts,
//   need, voff, num_runs, tstart
std::vector<at::Tensor> chained_plan(at::Tensor views, at::Tensor keys,
                                     at::Tensor tids, at::Tensor scores,
                                     int64_t total_buckets, int64_t num_tables) {
  check_cuda(keys, "keys");
  check_cuda(tids, "table ids");
  check_cuda(scores, "scores");
  const int64_t n = keys.numel();
  TORCH_CHECK(n > 0, "nothing to plan");
  auto o64 = keys.options().dtype(at::kLong);
  auto stream = at::cuda::getCurrentCUDAStream();

  auto [gid, local] = chained_gids(views, keys, tids);
  auto perm_in = at::arange(n, o64);
  auto sgid = at::empty({n}, o64);
  auto perm = at::empty({n}, o64);
  int bits = 1;
  while ((int64_t(1) << bits) < total_buckets) ++bits;
  size_t tb = 0;
  cub::DeviceRadixSort::SortPairs(nullptr, tb, gid.data_ptr<int64_t>(),
                                  sgid.data_ptr<int64_t>(),
                                  perm_in.data_ptr<int64_t>(),
                                  perm.data_ptr<int64_t>(), int(n), 0, bits,
                                  stream);
  {
    auto tmp = temp_storage(tb);
    cub::DeviceRadixSort::SortPairs(tmp.data_ptr(), tb, gid.data_ptr<int64_t>(),
                                    sgid.data_ptr<int64_t>(),
                                    perm_in.data_ptr<int64_t>(),
                                    perm.data_ptr<int64_t>(), int(n), 0, bits,
                                    stream);
  }
  auto skeys = at::empty({n}, o64);
  auto stids = at::empty({n}, o64);
  auto sscores = at::empty({n}, o64);
  auto slocal = at::empty({n}, o64);
  gather_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      perm.data_ptr<int64_t>(), reinterpret_cast<const int64_t *>(keys.data_ptr()),
      tids.data_ptr<int64_t>(),
      reinterpret_cast<const int64_t *>(scores.data_ptr()),
      local.data_ptr<int64_t>(), n, skeys.data_ptr<int64_t>(),
      stids.data_ptr<int64_t>(), sscores.data_ptr<int64_t>(),
      slocal.data_ptr<int64_t>());

  auto seg_gid = at::empty({n}, o64);
  auto counts = at::zeros({n}, o64);
  auto num_runs = at::zeros({1}, o64);
  tb = 0;
  cub::DeviceRunLengthEncode::Encode(nullptr, tb, sgid.data_ptr<int64_t>(),
                                     seg_gid.data_ptr<int64_t>(),
                                     counts.data_ptr<int64_t>(),
                                     num_runs.data_ptr<int64_t>(), int(n), stream);
  {
    auto tmp = temp_storage(tb);
    cub::DeviceRunLengthEncode::Encode(
        tmp.data_ptr(), tb, sgid.data_ptr<int64_t>(), seg_gid.data_ptr<int64_t>(),
        counts.data_ptr<int64_t>(), num_runs.data_ptr<int64_t>(), int(n), stream);
  }
  auto seg_begin = at::empty({n}, o64);
  tb = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, tb, counts.data_ptr<int64_t>(),
                                seg_begin.data_ptr<int64_t>(), int(n), stream);
  {
    auto tmp = temp_storage(tb);
    cub::DeviceScan::ExclusiveSum(tmp.data_ptr(), tb, counts.data_ptr<int64_t>(),
                                  seg_begin.data_ptr<int64_t>(), int(n), stream);
  }
  auto seg_table = at::empty({n}, o64);
  auto seg_bucket = at::empty({n}, o64);
  auto need = at::empty({n}, o64);
  segment_info_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      views_ptr(views), stids.data_ptr<int64_t>(), slocal.data_ptr<int64_t>(),
      seg_begin.data_ptr<int64_t>(), counts.data_ptr<int64_t>(),
      num_runs.data_ptr<int64_t>(), n, seg_table.data_ptr<int64_t>(),
      seg_bucket.data_ptr<int64_t>(), need.data_ptr<int64_t>());
  auto voff = at::empty({n}, o64);
  tb = 0;
  cub::DeviceScan::ExclusiveSum(nullptr, tb, need.data_ptr<int64_t>(),
                                voff.data_ptr<int64_t>(), int(n), stream);
  {
    auto tmp = temp_storage(tb);
    cub::DeviceScan::ExclusiveSum(tmp.data_ptr(), tb, need.data_ptr<int64_t>(),
                                  voff.data_ptr<int64_t>(), int(n), stream);
  }
  auto tstart = at::empty({num_tables + 1}, o64);
  table_start_kernel<<<grid_for(num_tables + 1, 128), 128, 0, stream>>>(
      views_ptr(views), sgid.data_ptr<int64_t>(), n, num_tables,
      tstart.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {perm,      sgid,   skeys, stids, sscores, seg_table, seg_bucket,
          seg_begin, counts, need,  voff,  num_runs, tstart};
}

// Take the rows, place the keys, put every key's row back in input order. Returns
//   rows (input order), srows (sorted order), vict_slot, vkeys, vrows, vtabs,
//   seg_fail, stats
// where stats (on the host) is [victims, runs, failed buckets, pool exhausted].
// If the pool is exhausted nothing is taken and nothing placed. One host
// synchronisation, for stats.
std::vector<at::Tensor> chained_apply(
    at::Tensor views, at::Tensor perm, at::Tensor skeys, at::Tensor stids,
    at::Tensor sscores, at::Tensor seg_table, at::Tensor seg_bucket,
    at::Tensor seg_begin, at::Tensor counts, at::Tensor need, at::Tensor voff,
    at::Tensor num_runs, at::Tensor tstart, int64_t guard_tables,
    int64_t guard_num_tables, bool pin_new, int64_t num_tables) {
  const int64_t n = perm.numel();
  auto o64 = perm.options();
  auto stream = at::cuda::getCurrentCUDAStream();
  auto abort = at::zeros({1}, o64.dtype(at::kInt));
  auto stats = at::zeros({4}, o64);
  auto srows = at::empty({n}, o64);
  auto rows = at::empty({n}, o64);
  auto vict_slot = at::empty({n}, o64.dtype(at::kInt));
  auto vkeys = at::empty({n}, o64);
  auto vrows = at::empty({n}, o64);
  auto vtabs = at::empty({n}, o64);
  auto seg_fail = at::zeros({n}, o64.dtype(at::kInt));
  embedding_cache::EvictionGuardView guard{
      reinterpret_cast<const void *>(guard_tables),
      static_cast<int32_t>(guard_num_tables)};
  const TableView *views_d = views_ptr(views);

  check_pool_kernel<<<1, 128, 0, stream>>>(views_d, tstart.data_ptr<int64_t>(),
                                           num_tables, abort.data_ptr<int32_t>());
  assign_rows_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      views_d, stids.data_ptr<int64_t>(), tstart.data_ptr<int64_t>(),
      abort.data_ptr<int32_t>(), n, srows.data_ptr<int64_t>());
  pop_rows_kernel<<<1, 128, 0, stream>>>(views_d, tstart.data_ptr<int64_t>(),
                                         num_tables, abort.data_ptr<int32_t>());
  place_kernel<<<grid_for(n, kWarpsPerBlock), kWarpsPerBlock * 32, 0, stream>>>(
      views_d, reinterpret_cast<const int64_t *>(skeys.data_ptr()),
      reinterpret_cast<const uint64_t *>(sscores.data_ptr()),
      srows.data_ptr<int64_t>(), seg_table.data_ptr<int64_t>(),
      seg_bucket.data_ptr<int64_t>(), seg_begin.data_ptr<int64_t>(),
      counts.data_ptr<int64_t>(), voff.data_ptr<int64_t>(),
      num_runs.data_ptr<int64_t>(), abort.data_ptr<int32_t>(),
      vict_slot.data_ptr<int32_t>(), vkeys.data_ptr<int64_t>(),
      vrows.data_ptr<int64_t>(), vtabs.data_ptr<int64_t>(),
      seg_fail.data_ptr<int32_t>(),
      reinterpret_cast<unsigned long long *>(stats.data_ptr<int64_t>() + 2),
      guard, pin_new ? 1 : 0);
  unpermute_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      perm.data_ptr<int64_t>(), srows.data_ptr<int64_t>(), n,
      rows.data_ptr<int64_t>());
  stats_kernel<<<1, 1, 0, stream>>>(voff.data_ptr<int64_t>(),
                                    need.data_ptr<int64_t>(),
                                    num_runs.data_ptr<int64_t>(), n,
                                    abort.data_ptr<int32_t>(),
                                    stats.data_ptr<int64_t>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto host = stats.to(at::kCPU);  // the one synchronisation
  return {rows, srows, vict_slot, vkeys, vrows, vtabs, seg_fail, host};
}

// Put rows back on a table's stack, in the order given.
void chained_release(at::Tensor views, int64_t table, at::Tensor rows) {
  check_cuda(rows, "rows");
  TORCH_CHECK(rows.scalar_type() == at::kLong, "rows must be int64");
  const int64_t n = rows.numel();
  if (n == 0) return;
  auto stream = at::cuda::getCurrentCUDAStream();
  release_kernel<<<grid_for(n, 256), 256, 0, stream>>>(
      views_ptr(views), table, rows.data_ptr<int64_t>(), n);
  push_rows_kernel<<<1, 1, 0, stream>>>(views_ptr(views), table, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace chained
} // namespace dyn_emb

void bind_chained_table(py::module &m) {
  m.def("chained_make_views", &dyn_emb::chained::chained_make_views,
        "build the device array of table views", py::arg("keys"),
        py::arg("scores"), py::arg("index"), py::arg("next"), py::arg("head"),
        py::arg("bucket_size"), py::arg("refs"), py::arg("free_rows"), py::arg("tops"),
        py::arg("num_buckets"), py::arg("bucket_capacity"));
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
  m.def("chained_stage", &dyn_emb::chained::chained_stage,
        "compact the misses after chained_find: (count, keys, table ids, scores, "
        "positions), each sized to the input",
        py::arg("rows"), py::arg("keys"), py::arg("table_ids"),
        py::arg("scores") = py::none(), py::arg("fill") = 0);
  m.def("chained_plan", &dyn_emb::chained::chained_plan,
        "sort staged keys by bucket and cut them into one run per bucket",
        py::arg("views"), py::arg("keys"), py::arg("table_ids"),
        py::arg("scores"), py::arg("total_buckets"), py::arg("num_tables"));
  m.def("chained_apply", &dyn_emb::chained::chained_apply,
        "take the rows, place the planned keys, return rows in input order",
        py::arg("views"), py::arg("perm"), py::arg("skeys"), py::arg("stids"),
        py::arg("sscores"), py::arg("seg_table"), py::arg("seg_bucket"),
        py::arg("seg_begin"), py::arg("counts"), py::arg("need"),
        py::arg("voff"), py::arg("num_runs"), py::arg("tstart"),
        py::arg("guard_tables"), py::arg("guard_num_tables"),
        py::arg("pin_new"), py::arg("num_tables"));
  m.def("chained_release", &dyn_emb::chained::chained_release,
        "put rows back on a table's stack", py::arg("views"), py::arg("table"),
        py::arg("rows"));
}
