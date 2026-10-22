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
// Several tables, one kernel.
//
// Each logical table owns a complete, independent set of structures — row pool,
// node metadata, buckets, ring. Nothing is shared. Three things fall out of
// that, and one cost:
//
//   + row width is per table, so a dim-32 table does not pay for a dim-256
//     neighbour. A single shared pool would have to be as wide as the widest
//     table and every narrow row would waste the difference.
//   + buckets are per table, so a chain can only contain keys of that table and
//     no table id has to be stored in the node or compared during a walk.
//   + ring counters split T ways. With one pool every miss in a step contends
//     on a single atomicAdd(head); T pools spread that out.
//   - capacity cannot move between tables. A hot table cannot borrow from an
//     idle one, so the split has to be planned rather than discovered.
//
// ---------------------------------------------------------------------------
// Getting a thread to its table
// ---------------------------------------------------------------------------
//
// Input keys arrive grouped by table, with `table_range[t]` the first index
// belonging to table t (what segmented_unique already produces). So the mapping
// from element to table is a binary search — but doing one per thread would pay
// for information almost every thread shares with its neighbours.
//
// Instead each block checks whether its whole chunk lies in one table, which is
// true for all but at most T-1 blocks. When it does, thread 0 resolves the
// table once and publishes the view through shared memory, and the rest of the
// block reads it with no search and no repeated pointer chasing. Only blocks
// straddling a boundary fall back to per-thread search.
//
// The compacted miss array is a different matter: it is no longer contiguous by
// table, so rebuilding a range for it would need a scan. Each AllocSlot carries
// its table id instead — four bytes on an array sized by misses, not by keys.
//
// VERIFICATION STATUS
//   verified   — dispatch (including boundary-straddling blocks), per-table
//                isolation, mixed row widths, via test/lw_cache/run_host_tests.sh
//   unverified — the shared-memory fast path with a real block, and everything
//                else concurrency. Needs a GPU.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "cache.h"
#include "ops.h"

namespace lw_cache {

// ---------------------------------------------------------------------------
// Device view
// ---------------------------------------------------------------------------

template <typename ValueT>
struct MultiCacheView {
  using value_t = ValueT;

  // One entry per table. Kept as separate arrays rather than an array of
  // CacheView so the fast path loads only what it needs.
  ValueT** rows;
  StateWord** states;
  Key** keys;
  int32_t** prev;
  int32_t** next;
  int32_t** heads;
  int32_t** ring_slots;
  uint64_t** ring_head;
  uint64_t** ring_tail;
  int64_t* capacity;
  int64_t* num_buckets;
  int64_t* ring_size;
  int32_t* row_elems;
  MetricsView metrics;

  // CLOCK only; null when disabled.
  int32_t** retry_nodes;
  uint32_t* retry_count;
  int32_t* retry_capacity;

  int num_tables;

  // Device only, and deliberately so: every field it reads is a device pointer
  // array, so calling this from the host segfaults rather than failing
  // gracefully. Host code wants MultiCache::table(t) instead, which returns the
  // owning Cache. Marking it __device__ turns the mistake into a compile error.
  __device__ CacheView<ValueT> table(int t) const {
    CacheView<ValueT> v;
    v.rows = rows[t];
    v.states = states[t];
    v.chain = HashChainView{heads[t],       prev[t],        next[t],
                            keys[t],        num_buckets[t], num_buckets[t] - 1};
    v.ring = RingView{ring_slots[t], ring_head[t],     ring_tail[t],
                      ring_size[t],  ring_size[t] - 1, metrics};
    v.capacity = capacity[t];
    v.row_elems = row_elems[t];
    v.metrics = metrics;
    return v;
  }

  __device__ RetryBuffer retry(int t) const {
    if (retry_nodes == nullptr) return RetryBuffer{nullptr, nullptr, 0};
    return RetryBuffer{retry_nodes[t], &retry_count[t], retry_capacity[t]};
  }
};

// One backing per table.
//
// Not a convenience: per-table row widths are the main reason the structures are
// split at all, and a backing has a single row width, so a shared one would
// force every table back to a common width and give the saving straight back.
// Tables that do happen to share a width may point at the same underlying
// backing; that is the caller's choice, not a constraint here.
template <typename BackingViewT>
struct MultiBackingView {
  BackingViewT* views;  // device array, one per table

  __host__ __device__ BackingViewT& table(int t) const { return views[t]; }
};

// ---------------------------------------------------------------------------
// Element -> table
// ---------------------------------------------------------------------------

// `table_range` has num_tables + 1 entries; element i belongs to the table t
// with table_range[t] <= i < table_range[t+1].
__host__ __device__ inline int find_table(const int64_t* table_range,
                                          int num_tables, int64_t i) {
  int lo = 0, hi = num_tables;
  while (lo + 1 < hi) {
    const int mid = (lo + hi) >> 1;
    if (table_range[mid] <= i) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return lo;
}

// Runs `body(i, table, view)` for every element, resolving the table once per
// block where possible.
//
// `body` must tolerate being called from a block where only some lanes have
// work; it is only invoked for i < n.
template <typename ValueT, typename Body>
__device__ inline void dispatch_by_table(const MultiCacheView<ValueT>& mv,
                                         const int64_t* table_range, int64_t n,
                                         Body body) {
  __shared__ int s_table;
  __shared__ CacheView<ValueT> s_view;

  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    if (threadIdx.x == 0) {
      int64_t last = base + blockDim.x - 1;
      if (last > n - 1) last = n - 1;
      const int t0 = find_table(table_range, mv.num_tables, base);
      const int t1 = find_table(table_range, mv.num_tables, last);
      // -1 marks a block straddling a table boundary: at most T-1 of them.
      s_table = (t0 == t1) ? t0 : -1;
      if (s_table >= 0) s_view = mv.table(s_table);
    }
    __syncthreads();

    const int64_t i = base + threadIdx.x;
    if (i < n) {
      if (s_table >= 0) {
        body(i, s_table, s_view);
      } else {
        const int t = find_table(table_range, mv.num_tables, i);
        body(i, t, mv.table(t));
      }
    }
    // The next iteration overwrites the shared view, so nobody may still be
    // reading it.
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// Host owner
// ---------------------------------------------------------------------------

template <typename ValueT>
class MultiCache {
 public:
  using value_t = ValueT;
  using View = MultiCacheView<ValueT>;

  explicit MultiCache(const std::vector<CacheConfig>& configs) {
    if (configs.empty()) {
      throw std::invalid_argument("lw_cache: need at least one table");
    }
    const bool clock = configs[0].replacement == ReplacementPolicy::kClock;
    for (const auto& c : configs) {
      if ((c.replacement == ReplacementPolicy::kClock) != clock) {
        // Mixing policies is nearly free to implement, but it multiplies the
        // space a hit-rate regression has to be searched in by 2^T for no use
        // case anyone has asked for.
        throw std::invalid_argument(
            "lw_cache: all tables must share one replacement policy");
      }
      tables_.push_back(std::make_unique<Cache<ValueT>>(c));
    }
    clock_ = clock;
    build_device_arrays();
  }

  MultiCache(const MultiCache&) = delete;
  MultiCache& operator=(const MultiCache&) = delete;

  int num_tables() const { return static_cast<int>(tables_.size()); }
  Cache<ValueT>& table(int t) { return *tables_[static_cast<size_t>(t)]; }
  const Cache<ValueT>& table(int t) const {
    return *tables_[static_cast<size_t>(t)];
  }
  bool clock_enabled() const { return clock_; }

  // Counters are shared across tables: the questions they answer — hit rate,
  // whether CLOCK would pay, whether the ring ever ran dry — are about the
  // cache as a whole. Per-table breakdowns would need T times the counters and
  // nobody has asked for one.
  View view(MetricsView metrics = MetricsView{nullptr}) const {
    View v;
    v.metrics = metrics;
    v.rows = d_rows_;
    v.states = d_states_;
    v.keys = d_keys_;
    v.prev = d_prev_;
    v.next = d_next_;
    v.heads = d_heads_;
    v.ring_slots = d_ring_slots_;
    v.ring_head = d_ring_head_;
    v.ring_tail = d_ring_tail_;
    v.capacity = d_capacity_;
    v.num_buckets = d_num_buckets_;
    v.ring_size = d_ring_size_;
    v.row_elems = d_row_elems_;
    v.retry_nodes = clock_ ? d_retry_nodes_ : nullptr;
    v.retry_count = clock_ ? d_retry_count_ : nullptr;
    v.retry_capacity = clock_ ? d_retry_capacity_ : nullptr;
    v.num_tables = num_tables();
    return v;
  }

  void reset() {
    for (auto& t : tables_) t->reset();
    // The device arrays hold pointers into each table's own allocations, which
    // reset() rewrites in place. Nothing here goes stale.
  }

  size_t metadata_bytes() const {
    size_t n = 0;
    for (const auto& t : tables_) n += t->metadata_bytes();
    return n;
  }
  size_t row_bytes() const {
    size_t n = 0;
    for (const auto& t : tables_) n += t->row_bytes();
    return n;
  }

  ~MultiCache() {
    cudaFree(d_rows_);
    cudaFree(d_states_);
    cudaFree(d_keys_);
    cudaFree(d_prev_);
    cudaFree(d_next_);
    cudaFree(d_heads_);
    cudaFree(d_ring_slots_);
    cudaFree(d_ring_head_);
    cudaFree(d_ring_tail_);
    cudaFree(d_capacity_);
    cudaFree(d_num_buckets_);
    cudaFree(d_ring_size_);
    cudaFree(d_row_elems_);
    cudaFree(d_retry_nodes_);
    cudaFree(d_retry_count_);
    cudaFree(d_retry_capacity_);
  }

 private:
  // The per-table views are built once and uploaded; a kernel then reaches a
  // table with one indexed load per field instead of rebuilding anything.
  //
  // Ring head/tail are stored as *pointers* to each table's own counters, not
  // as copies of their values. Copies would give every table two counters, only
  // one of which reset() rewrites — and the stale pair would then bound the pop
  // loop. One authoritative location per table, by construction.
  void build_device_arrays() {
    const int T = num_tables();
    std::vector<ValueT*> rows(T);
    std::vector<StateWord*> states(T);
    std::vector<Key*> keys(T);
    std::vector<int32_t*> prev(T), next(T), heads(T), ring_slots(T);
    std::vector<uint64_t*> ring_head(T), ring_tail(T);
    std::vector<int64_t> capacity(T), num_buckets(T), ring_size(T);
    std::vector<int32_t> row_elems(T);
    std::vector<int32_t*> retry_nodes(T);
    std::vector<uint32_t> retry_count(T, 0u);
    std::vector<int32_t> retry_capacity(T, 0);

    for (int t = 0; t < T; ++t) {
      const auto v = tables_[static_cast<size_t>(t)]->view();
      rows[t] = v.rows;
      states[t] = v.states;
      keys[t] = v.chain.keys;
      prev[t] = v.chain.prev;
      next[t] = v.chain.next;
      heads[t] = v.chain.heads;
      ring_slots[t] = v.ring.slots;
      ring_head[t] = v.ring.head;
      ring_tail[t] = v.ring.tail;
      capacity[t] = v.capacity;
      num_buckets[t] = v.chain.num_buckets;
      ring_size[t] = v.ring.capacity;
      row_elems[t] = v.row_elems;
      const auto rb = tables_[static_cast<size_t>(t)]->retry_buffer();
      retry_nodes[t] = rb.nodes;
      retry_capacity[t] = rb.capacity;
    }

    upload(&d_rows_, rows);
    upload(&d_states_, states);
    upload(&d_keys_, keys);
    upload(&d_prev_, prev);
    upload(&d_next_, next);
    upload(&d_heads_, heads);
    upload(&d_ring_slots_, ring_slots);
    upload(&d_ring_head_, ring_head);
    upload(&d_ring_tail_, ring_tail);
    upload(&d_capacity_, capacity);
    upload(&d_num_buckets_, num_buckets);
    upload(&d_ring_size_, ring_size);
    upload(&d_row_elems_, row_elems);
    if (clock_) {
      upload(&d_retry_nodes_, retry_nodes);
      upload(&d_retry_count_, retry_count);
      upload(&d_retry_capacity_, retry_capacity);
    }
  }

  template <typename T>
  static void upload(T** dst, const std::vector<T>& src) {
    const size_t bytes = src.size() * sizeof(T);
    cudaError_t e = cudaMalloc(dst, bytes);
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("lw_cache: cudaMalloc: ") +
                               cudaGetErrorString(e));
    }
    e = cudaMemcpy(*dst, src.data(), bytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("lw_cache: cudaMemcpy: ") +
                               cudaGetErrorString(e));
    }
  }

  std::vector<std::unique_ptr<Cache<ValueT>>> tables_;
  bool clock_ = false;

  ValueT** d_rows_ = nullptr;
  StateWord** d_states_ = nullptr;
  Key** d_keys_ = nullptr;
  int32_t** d_prev_ = nullptr;
  int32_t** d_next_ = nullptr;
  int32_t** d_heads_ = nullptr;
  int32_t** d_ring_slots_ = nullptr;
  uint64_t** d_ring_head_ = nullptr;
  uint64_t** d_ring_tail_ = nullptr;
  int64_t* d_capacity_ = nullptr;
  int64_t* d_num_buckets_ = nullptr;
  int64_t* d_ring_size_ = nullptr;
  int32_t* d_row_elems_ = nullptr;
  int32_t** d_retry_nodes_ = nullptr;
  uint32_t* d_retry_count_ = nullptr;
  int32_t* d_retry_capacity_ = nullptr;
};

}  // namespace lw_cache
