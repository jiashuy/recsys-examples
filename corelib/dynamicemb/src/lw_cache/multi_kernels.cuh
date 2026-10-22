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
// Multi-table kernels.
//
// Every table goes through one launch. The per-key work is unchanged — these
// resolve which table an element belongs to and then call the same operations
// the single-table kernels do.
//
// Two dispatch styles, because the two input shapes differ:
//
//   keyed by table_range   lookup, release. Input is grouped by table, so a
//                          block usually spans one table and can resolve it
//                          once into shared memory.
//   keyed by slot.table    the allocate passes. The compacted miss array is no
//                          longer contiguous by table, and rebuilding a range
//                          for it would need a scan; four bytes per slot is
//                          cheaper.
//
// Same phase boundaries as the single-table path, for the same reasons; see
// ops.h.
//
// VERIFICATION STATUS
//   verified   — kernel bodies including boundary-straddling dispatch, via the
//                host shim at one thread per grid
//   unverified — the shared-memory fast path with a real block, launch
//                configuration, all concurrency. Needs a GPU.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "kernels.cuh"
#include "multi_cache.h"

namespace lw_cache {

// ---------------------------------------------------------------------------
// Pin phase
// ---------------------------------------------------------------------------

// `keys` must be deduplicated within each table and grouped by table, which is
// what segmented_unique produces.
template <typename ValueT>
__global__ void mt_lookup_pin_kernel(MultiCacheView<ValueT> mv, const Key* keys,
                                     const int64_t* table_range,
                                     int64_t num_keys, int32_t* out_nodes) {
  dispatch_by_table(mv, table_range, num_keys,
                    [&] __device__(int64_t i, int, const CacheView<ValueT>& v) {
                      out_nodes[i] = cache_lookup_pin(v, keys[i]);
                    });
}

// Multi-table variant: also resolves each miss's table from `table_range`.
template <typename ValueT>
__global__ void mt_compact_misses_kernel(const int32_t* nodes, const Key* keys,
                                         const int64_t* table_range,
                                         int num_tables, int64_t n,
                                         AllocSlot* slots, uint32_t* count) {
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    if (nodes[i] != kNoNode) continue;
    const uint32_t at = atomicAdd(count, 1u);
    AllocSlot s;
    s.key = keys[i];
    s.src = static_cast<int32_t>(i);
    s.table = static_cast<int32_t>(find_table(table_range, num_tables, i));
    slots[at] = s;
  }
}

// ---------------------------------------------------------------------------
// Allocate
// ---------------------------------------------------------------------------

// The caller fills slots[j].key and slots[j].table. Tail snapshots are read per
// table before the launch, for the same reason as the single-table path: pushes
// cannot happen while this runs, so each bound is constant.
template <typename ValueT, typename BackingT>
__global__ void mt_claim_kernel(MultiCacheView<ValueT> mv,
                                MultiBackingView<BackingT> backing,
                                AllocSlot* slots, int64_t num_slots,
                                const uint64_t* tail_snapshots, PopPolicy policy,
                                bool use_retry) {
  for (int64_t i = global_thread_id(); i < num_slots; i += thread_stride()) {
    const int t = slots[i].table;
    const CacheView<ValueT> v = mv.table(t);
    RetryBuffer rb = mv.retry(t);
    cache_claim(v, backing.table(t), slots[i], tail_snapshots[t], policy,
                use_retry ? &rb : nullptr);
  }
}

// One thread per table: each drains its own retry buffer. The counts are small
// (bounded by that table's misses) and the tables are independent, so there is
// nothing to gain from a finer split.
template <typename ValueT>
__global__ void mt_absorb_retry_kernel(MultiCacheView<ValueT> mv) {
  for (int64_t t = global_thread_id(); t < mv.num_tables; t += thread_stride()) {
    const CacheView<ValueT> v = mv.table(static_cast<int>(t));
    const RetryBuffer rb = mv.retry(static_cast<int>(t));
    const uint32_t n = *rb.count;
    for (uint32_t i = 0; i < n; ++i) ring_absorb_retry(v.ring, rb, i);
    *rb.count = 0;
  }
}

template <typename ValueT, typename BackingT>
__global__ void mt_unlink_kernel(MultiCacheView<ValueT> mv,
                                 MultiBackingView<BackingT> backing,
                                 AllocSlot* slots, int64_t num_slots) {
  for (int64_t i = global_thread_id(); i < num_slots; i += thread_stride()) {
    cache_unlink_victim(mv.table(slots[i].table), backing.table(slots[i].table),
                        slots[i]);
  }
}

template <typename ValueT>
__global__ void mt_bind_kernel(MultiCacheView<ValueT> mv, AllocSlot* slots,
                               int64_t num_slots) {
  for (int64_t i = global_thread_id(); i < num_slots; i += thread_stride()) {
    cache_bind(mv.table(slots[i].table), slots[i]);
  }
}

template <int kLanes, typename ValueT, typename BackingT>
__global__ void mt_writeback_kernel(MultiCacheView<ValueT> mv,
                                    MultiBackingView<BackingT> backing,
                                    const AllocSlot* slots, int64_t num_slots) {
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t s = tid / kLanes; s < num_slots; s += thread_stride() / kLanes) {
    cache_writeback(mv.table(slots[s].table), backing.table(slots[s].table),
                    slots[s], lane, kLanes);
  }
}

template <int kLanes, typename ValueT, typename BackingT, typename Init>
__global__ void mt_fill_kernel(MultiCacheView<ValueT> mv,
                               MultiBackingView<BackingT> backing,
                               const AllocSlot* slots, int64_t num_slots,
                               Init init) {
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t s = tid / kLanes; s < num_slots; s += thread_stride() / kLanes) {
    const CacheView<ValueT> v = mv.table(slots[s].table);
    cache_fill(v, backing.table(slots[s].table), slots[s], init, lane, kLanes);
    __syncwarp();
    if (lane == 0) cache_publish(v, slots[s]);
  }
}

// ---------------------------------------------------------------------------
// Device-count variants
// ---------------------------------------------------------------------------

template <typename ValueT, typename BackingT>
__global__ void mt_claim_kernel_n(MultiCacheView<ValueT> mv,
                                  MultiBackingView<BackingT> backing,
                                  AllocSlot* slots, const uint32_t* count,
                                  const uint64_t* tail_snapshots,
                                  PopPolicy policy, bool use_retry) {
  const int64_t n = static_cast<int64_t>(*count);
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    const int t = slots[i].table;
    RetryBuffer rb = mv.retry(t);
    cache_claim(mv.table(t), backing.table(t), slots[i], tail_snapshots[t],
                policy, use_retry ? &rb : nullptr);
  }
}

template <typename ValueT, typename BackingT>
__global__ void mt_unlink_kernel_n(MultiCacheView<ValueT> mv,
                                   MultiBackingView<BackingT> backing,
                                   AllocSlot* slots, const uint32_t* count) {
  const int64_t n = static_cast<int64_t>(*count);
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    cache_unlink_victim(mv.table(slots[i].table), backing.table(slots[i].table),
                        slots[i]);
  }
}

template <typename ValueT>
__global__ void mt_bind_kernel_n(MultiCacheView<ValueT> mv, AllocSlot* slots,
                                 const uint32_t* count) {
  const int64_t n = static_cast<int64_t>(*count);
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    cache_bind(mv.table(slots[i].table), slots[i]);
  }
}

template <int kLanes, typename ValueT, typename BackingT>
__global__ void mt_writeback_kernel_n(MultiCacheView<ValueT> mv,
                                      MultiBackingView<BackingT> backing,
                                      const AllocSlot* slots,
                                      const uint32_t* count) {
  const int64_t n = static_cast<int64_t>(*count);
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t s = tid / kLanes; s < n; s += thread_stride() / kLanes) {
    cache_writeback(mv.table(slots[s].table), backing.table(slots[s].table),
                    slots[s], lane, kLanes);
  }
}

template <int kLanes, typename ValueT, typename BackingT, typename Init>
__global__ void mt_fill_kernel_n(MultiCacheView<ValueT> mv,
                                 MultiBackingView<BackingT> backing,
                                 const AllocSlot* slots, const uint32_t* count,
                                 Init init) {
  const int64_t n = static_cast<int64_t>(*count);
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t s = tid / kLanes; s < n; s += thread_stride() / kLanes) {
    const CacheView<ValueT> v = mv.table(slots[s].table);
    cache_fill(v, backing.table(slots[s].table), slots[s], init, lane, kLanes);
    __syncwarp();
    if (lane == 0) cache_publish(v, slots[s]);
  }
}

// ---------------------------------------------------------------------------
// Release
// ---------------------------------------------------------------------------

// Node ids alone do not say which table they belong to — ids are per table and
// therefore ambiguous across tables. They are dispatched by table_range, in the
// same order as the keys that produced them.
template <typename ValueT>
__global__ void mt_release_kernel(MultiCacheView<ValueT> mv,
                                  const int32_t* nodes,
                                  const int64_t* table_range, int64_t num_nodes) {
  dispatch_by_table(mv, table_range, num_nodes,
                    [&] __device__(int64_t i, int, const CacheView<ValueT>& v) {
                      cache_release(v, nodes[i]);
                    });
}

// ---------------------------------------------------------------------------
// Flush and sweep
// ---------------------------------------------------------------------------

// One launch over every node of every table. `node_range[t]` gives the first
// flat index belonging to table t, so the same dispatch applies.
template <int kLanes, typename ValueT, typename BackingT>
__global__ void mt_flush_kernel(MultiCacheView<ValueT> mv,
                                MultiBackingView<BackingT> backing,
                                const int64_t* node_range, int64_t total_nodes) {
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t g = tid / kLanes; g < total_nodes;
       g += thread_stride() / kLanes) {
    const int t = find_table(node_range, mv.num_tables, g);
    const CacheView<ValueT> v = mv.table(t);
    const int32_t node = static_cast<int32_t>(g - node_range[t]);
    cache_flush_node(v, backing.table(t), node, lane, kLanes);
  }
}

template <typename ValueT>
__global__ void mt_count_referenced_kernel(MultiCacheView<ValueT> mv,
                                           const int64_t* node_range,
                                           int64_t total_nodes,
                                           unsigned long long* out) {
  for (int64_t g = global_thread_id(); g < total_nodes; g += thread_stride()) {
    const int t = find_table(node_range, mv.num_tables, g);
    const CacheView<ValueT> v = mv.table(t);
    const int64_t node = g - node_range[t];
    if (ref_of(v.states[node]) != 0u) atomicAdd(out, 1ull);
  }
}

}  // namespace lw_cache
