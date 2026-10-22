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
// Cache storage: the row pool, the node metadata, and the two structures that
// index it.
//
// Per node, 32 bytes of metadata:
//
//   key         8      identity, and the bucket a victim must leave
//   state       4      the whole concurrency protocol (node_state.h)
//   prev/next   8      bucket chain
//   heads       8      2 buckets per node, 4 bytes each
//   ring        4      one slot per node
//
// against a row of, say, 1536 B for dim 128 in fp32 with Adam — about 2%. The
// interesting memory question is not this; it is the row width itself.
//
// Note what is *not* here: no backing row index. A victim's row is resolved by
// probing the backing at eviction time. That costs one probe and buys immunity
// to backing rehash and to backing eviction — a stored row index could go stale
// under either, and a stale row index means writing a row on top of some other
// key's data, silently.
//
// VERIFICATION STATUS
//   verified   — sizing, capacity validation, initial state of every array,
//                via testing/run_host_tests.sh
//   unverified — nothing device-specific here; allocation and upload are the
//                only CUDA calls and they are one-time.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "backing.h"
#include "hash_chain.h"
#include "metrics.h"
#include "node_state.h"
#include "ring.h"

namespace lw_cache {

enum class ReplacementPolicy { kFifo, kClock };

// A node's key field is only meaningful once the node is bound to a key. EMPTY
// nodes sit in no chain, so this is never compared against a real key; it exists
// so that a node that *is* wrongly compared stands out.  Callers must not use it
// as a real key.
constexpr Key kUnboundKey = INT64_MIN;

struct CacheConfig {
  int64_t capacity = 0;       // nodes
  int row_elems = 0;          // value_t per row: emb dims + optimizer state dims

  ReplacementPolicy replacement = ReplacementPolicy::kFifo;
  int clock_max_second_chances = 1;
  int prefetch_depth = 1;

  // Used only to validate capacity at construction. Zero disables the check,
  // which is what the unit tests want and what production should not do.
  int64_t expected_unique_per_step = 0;
  double capacity_safety_factor = 1.5;

  PopPolicy pop_policy() const {
    return replacement == ReplacementPolicy::kClock
               ? PopPolicy::clock_with(clock_max_second_chances)
               : PopPolicy::fifo();
  }
};

// ---------------------------------------------------------------------------
// Device view
// ---------------------------------------------------------------------------

template <typename ValueT>
struct CacheView {
  using value_t = ValueT;

  value_t* rows;         // [capacity * row_elems]
  StateWord* states;     // [capacity]
  HashChainView chain;   // keys / prev / next / heads
  RingView ring;
  int64_t capacity;
  int row_elems;
  MetricsView metrics;

  __host__ __device__ value_t* row(int32_t node) const {
    return rows + static_cast<int64_t>(node) * static_cast<int64_t>(row_elems);
  }
  __host__ __device__ Key key_of(int32_t node) const { return chain.keys[node]; }
};

// ---------------------------------------------------------------------------
// Host owner
// ---------------------------------------------------------------------------

template <typename ValueT>
class Cache {
 public:
  using value_t = ValueT;
  using View = CacheView<ValueT>;

  explicit Cache(const CacheConfig& cfg) : cfg_(cfg) {
    validate(cfg);

    capacity_ = cfg.capacity;
    row_elems_ = cfg.row_elems;
    num_buckets_ = round_up_pow2(2 * capacity_);
    // The ring is sized independently of the node count: its mask needs a power
    // of two, and rounding the node count itself would nearly double the row
    // pool, which is the one array where a factor of two actually costs
    // something. Live entries never exceed `capacity` (INV-4), so a larger
    // slot array is simply never full.
    ring_slots_ = round_up_pow2(capacity_);

    alloc(&d_rows_, static_cast<size_t>(capacity_) *
                        static_cast<size_t>(row_elems_) * sizeof(value_t),
          "rows");
    alloc(&d_states_, static_cast<size_t>(capacity_) * sizeof(StateWord), "states");
    alloc(&d_keys_, static_cast<size_t>(capacity_) * sizeof(Key), "keys");
    alloc(&d_prev_, static_cast<size_t>(capacity_) * sizeof(int32_t), "prev");
    alloc(&d_next_, static_cast<size_t>(capacity_) * sizeof(int32_t), "next");
    alloc(&d_heads_, static_cast<size_t>(num_buckets_) * sizeof(int32_t), "heads");
    alloc(&d_ring_, static_cast<size_t>(ring_slots_) * sizeof(int32_t), "ring");
    alloc(&d_head_, sizeof(uint64_t), "ring head");
    alloc(&d_tail_, sizeof(uint64_t), "ring tail");

    if (cfg.replacement == ReplacementPolicy::kClock) {
      alloc(&d_retry_, static_cast<size_t>(capacity_) * sizeof(int32_t), "retry");
      alloc(&d_retry_count_, sizeof(uint32_t), "retry count");
    }

    reset();
  }

  ~Cache() {
    cudaFree(d_rows_);
    cudaFree(d_states_);
    cudaFree(d_keys_);
    cudaFree(d_prev_);
    cudaFree(d_next_);
    cudaFree(d_heads_);
    cudaFree(d_ring_);
    cudaFree(d_head_);
    cudaFree(d_tail_);
    cudaFree(d_retry_);
    cudaFree(d_retry_count_);
  }

  Cache(const Cache&) = delete;
  Cache& operator=(const Cache&) = delete;

  // Bring every node back to EMPTY and refill the ring with arange(capacity).
  //
  // Done on the host and uploaded: it happens once per construction (and after
  // a full flush), so there is nothing to gain from a kernel, and a host loop
  // is testable without a GPU.
  void reset() {
    const size_t cap = static_cast<size_t>(capacity_);

    std::vector<StateWord> states(cap, initial_state());
    upload(d_states_, states, "states");

    // EMPTY nodes are in no chain, so their keys are never compared. Fill with
    // a value that stands out if one ever is.
    std::vector<Key> keys(cap, kUnboundKey);
    upload(d_keys_, keys, "keys");

    std::vector<int32_t> nil(cap, kNoNode);
    upload(d_prev_, nil, "prev");
    upload(d_next_, nil, "next");

    std::vector<int32_t> heads(static_cast<size_t>(num_buckets_), kNoNode);
    upload(d_heads_, heads, "heads");

    // arange: every node starts in the ring, matching in_ring set in
    // initial_state(). This is what collapses "never used" and "used and
    // released" into a single pool.
    std::vector<int32_t> ring(static_cast<size_t>(ring_slots_), kNoNode);
    for (int64_t i = 0; i < capacity_; ++i) ring[static_cast<size_t>(i)] = int32_t(i);
    upload(d_ring_, ring, "ring");

    const uint64_t head = 0, tail = static_cast<uint64_t>(capacity_);
    check(cudaMemcpy(d_head_, &head, sizeof(head), cudaMemcpyHostToDevice), "head");
    check(cudaMemcpy(d_tail_, &tail, sizeof(tail), cudaMemcpyHostToDevice), "tail");

    if (d_retry_count_) {
      const uint32_t zero = 0;
      check(cudaMemcpy(d_retry_count_, &zero, sizeof(zero), cudaMemcpyHostToDevice),
            "retry count");
    }

    // Rows are left untouched: an EMPTY node's row is never read, and zeroing
    // the whole pool would be the single largest write in construction.
  }

  // `metrics` may be a disabled view; every bump then compiles to an untaken,
  // warp-uniform branch.
  View view(MetricsView metrics = MetricsView{nullptr}) const {
    View v;
    v.metrics = metrics;
    v.rows = d_rows_;
    v.states = d_states_;
    v.chain = HashChainView{d_heads_, d_prev_,     d_next_,
                            d_keys_,  num_buckets_, num_buckets_ - 1};
    v.ring = RingView{d_ring_,     d_head_,     d_tail_,
                      ring_slots_, ring_slots_ - 1, metrics};
    v.capacity = capacity_;
    v.row_elems = row_elems_;
    return v;
  }

  RetryBuffer retry_buffer() const {
    return RetryBuffer{d_retry_, d_retry_count_, static_cast<int32_t>(capacity_)};
  }
  bool clock_enabled() const {
    return cfg_.replacement == ReplacementPolicy::kClock;
  }

  const CacheConfig& config() const { return cfg_; }
  int64_t capacity() const { return capacity_; }
  int row_elems() const { return row_elems_; }
  int64_t num_buckets() const { return num_buckets_; }
  int64_t ring_slots() const { return ring_slots_; }

  // Bytes of node metadata, excluding the row pool.
  size_t metadata_bytes() const {
    return static_cast<size_t>(capacity_) *
               (sizeof(StateWord) + sizeof(Key) + 2 * sizeof(int32_t)) +
           static_cast<size_t>(num_buckets_) * sizeof(int32_t) +
           static_cast<size_t>(ring_slots_) * sizeof(int32_t);
  }
  size_t row_bytes() const {
    return static_cast<size_t>(capacity_) * static_cast<size_t>(row_elems_) *
           sizeof(value_t);
  }

  // -- inspection, for tests and for the consistency sweep -----------------

  std::vector<StateWord> download_states() const {
    return download(d_states_, static_cast<size_t>(capacity_));
  }
  std::vector<Key> download_keys() const {
    return download(d_keys_, static_cast<size_t>(capacity_));
  }
  std::vector<int32_t> download_heads() const {
    return download(d_heads_, static_cast<size_t>(num_buckets_));
  }
  std::vector<int32_t> download_ring() const {
    return download(d_ring_, static_cast<size_t>(ring_slots_));
  }
  void download_ring_counters(uint64_t* head, uint64_t* tail) const {
    check(cudaMemcpy(head, d_head_, sizeof(*head), cudaMemcpyDeviceToHost), "head");
    check(cudaMemcpy(tail, d_tail_, sizeof(*tail), cudaMemcpyDeviceToHost), "tail");
  }

 private:
  static void validate(const CacheConfig& cfg) {
    if (cfg.capacity <= 0) {
      throw std::invalid_argument("lw_cache: capacity must be > 0");
    }
    if (cfg.row_elems <= 0) {
      throw std::invalid_argument("lw_cache: row_elems must be > 0");
    }
    if (cfg.prefetch_depth <= 0) {
      throw std::invalid_argument("lw_cache: prefetch_depth must be > 0");
    }
    if (cfg.capacity > static_cast<int64_t>(INT32_MAX)) {
      throw std::invalid_argument(
          "lw_cache: capacity exceeds the int32 node id range");
    }
    if (cfg.replacement == ReplacementPolicy::kClock &&
        cfg.clock_max_second_chances <= 0) {
      throw std::invalid_argument(
          "lw_cache: CLOCK needs clock_max_second_chances > 0; without a "
          "budget the pop loop can sweep every entry into the retry buffer and "
          "then wrongly report the ring exhausted");
    }

    // Every key touched by an in-flight step is pinned, and a pinned node
    // cannot be reclaimed. Undersize the cache and allocation fails mid-step
    // with the ring dry — a hard stop, not a slowdown. Catch it here instead.
    //
    // CLOCK needs more than FIFO, which is easy to miss. A thread granting a
    // second chance consumes a ring entry without getting a node for it: the
    // entry moves to the retry buffer and does not come back until the
    // follow-up kernel. So each thread can take up to `k + 1` entries, against
    // FIFO's 1. Sizing for FIFO and then switching policy is exactly how this
    // shows up — as a mid-run exhaustion, hundreds of steps in.
    if (cfg.expected_unique_per_step > 0) {
      const int per_thread =
          cfg.replacement == ReplacementPolicy::kClock
              ? 1 + cfg.clock_max_second_chances
              : 1;
      const double need = static_cast<double>(cfg.expected_unique_per_step) *
                          cfg.prefetch_depth * cfg.capacity_safety_factor *
                          per_thread;
      if (static_cast<double>(cfg.capacity) < need) {
        throw std::invalid_argument(
            "lw_cache: capacity " + std::to_string(cfg.capacity) +
            " is below what a step can hold at once: " +
            std::to_string(cfg.expected_unique_per_step) + " unique keys x " +
            std::to_string(cfg.prefetch_depth) + " prefetch depth x " +
            std::to_string(cfg.capacity_safety_factor) + " safety x " +
            std::to_string(per_thread) +
            " ring entries per allocating thread (CLOCK spends one per second "
            "chance on top of the one it claims) = " +
            std::to_string(static_cast<int64_t>(need)));
      }
    }
  }

  static int64_t round_up_pow2(int64_t v) {
    int64_t p = 1;
    while (p < v) p <<= 1;
    return p;
  }

  static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("lw_cache: ") + what + ": " +
                               cudaGetErrorString(e));
    }
  }

  template <typename T>
  static void alloc(T** p, size_t bytes, const char* what) {
    check(cudaMalloc(p, bytes), what);
  }

  template <typename T>
  static void upload(T* dst, const std::vector<T>& src, const char* what) {
    check(cudaMemcpy(dst, src.data(), src.size() * sizeof(T),
                     cudaMemcpyHostToDevice),
          what);
  }

  template <typename T>
  static std::vector<T> download(const T* src, size_t n) {
    std::vector<T> out(n);
    check(cudaMemcpy(out.data(), src, n * sizeof(T), cudaMemcpyDeviceToHost),
          "download");
    return out;
  }

  CacheConfig cfg_;
  int64_t capacity_ = 0;
  int row_elems_ = 0;
  int64_t num_buckets_ = 0;
  int64_t ring_slots_ = 0;

  value_t* d_rows_ = nullptr;
  StateWord* d_states_ = nullptr;
  Key* d_keys_ = nullptr;
  int32_t* d_prev_ = nullptr;
  int32_t* d_next_ = nullptr;
  int32_t* d_heads_ = nullptr;
  int32_t* d_ring_ = nullptr;
  uint64_t* d_head_ = nullptr;
  uint64_t* d_tail_ = nullptr;
  int32_t* d_retry_ = nullptr;
  uint32_t* d_retry_count_ = nullptr;
};

}  // namespace lw_cache
