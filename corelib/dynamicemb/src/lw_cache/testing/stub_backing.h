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
// StubBacking — a minimal backing implementation used as a test fixture.
//
// NOT PRODUCTION CODE.  It exists so the cache can be developed and validated
// against a backing whose behaviour we fully control: we can force it to
// evict, force it to fill up, and read its entire contents to compare against
// a reference run (the differential test, see docs/lightweight_cache_design.md
// P0 acceptance).
//
// Layout: open-addressed hash table, linear probing, power-of-two slots.
// The slot index *is* the row index, so resolving a key and locating its row
// are the same operation.
//
// Two deliberate simplifications, both safe for a fixture:
//
//   * No tombstones.  Eviction is a host-side full rebuild (`evict_host`).
//     Besides being far simpler than tombstone reuse, this makes eviction move
//     every surviving key to a new slot — i.e. it exercises backing rehash,
//     which the cache claims immunity to precisely because it never stores a
//     row index across kernel boundaries.  A fixture that quietly kept rows in
//     place would let a violation of that rule pass unnoticed.
//
//   * Fixed capacity.  Filling up returns kRowAllocFailed, which is a path the
//     cache must handle anyway (non-admitted bypass).
//
// VERIFICATION STATUS
//   verified   — probe/insert/rebuild logic, via testing/run_host_tests.sh
//                (host-only shim, single-threaded)
//   unverified — everything device-specific: atomicCAS under contention, the
//                volatile load, mapped-pinned zero-copy addressing.  Needs a
//                CUDA toolchain; none on the current development machine.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../backing.h"
#include "../hash.h"

namespace lw_cache {
namespace testing {

// Slot sentinels.  Real keys must never take these values; `StubBacking`
// rejects them on insert so a test cannot silently corrupt the table.
constexpr Key kEmptyKey = INT64_MIN;

// Reads of `keys` race with concurrent inserts by other threads, so they must
// not be hoisted into a register or served from a stale cache line.
__device__ inline Key load_key(const Key* p) {
  return *reinterpret_cast<const volatile Key*>(p);
}

// ---------------------------------------------------------------------------
// Device view — satisfies the lw_cache backing contract.
// ---------------------------------------------------------------------------

template <typename ValueT>
struct StubBackingView {
  using value_t = ValueT;

  Key* keys;        // [slots], device memory, kEmptyKey = free
  value_t* rows;      // [slots * elems], device pointer into mapped pinned host
  int64_t slots;      // power of two
  int elems;          // value_t per row: emb dims + optimizer state dims

  __device__ Row probe(Key key) const {
    const int64_t mask = slots - 1;
    int64_t idx = static_cast<int64_t>(mix64(static_cast<uint64_t>(key)) &
                                       static_cast<uint64_t>(mask));
    for (int64_t i = 0; i < slots; ++i) {
      const Key k = load_key(&keys[idx]);
      if (k == key) return idx;
      // An empty slot ends the probe sequence: had the key been inserted, it
      // would sit at or before this point.
      if (k == kEmptyKey) return kRowNotFound;
      idx = (idx + 1) & mask;
    }
    return kRowNotFound;
  }

  __device__ Row find_or_insert(Key key) {
    const int64_t mask = slots - 1;
    int64_t idx = static_cast<int64_t>(mix64(static_cast<uint64_t>(key)) &
                                       static_cast<uint64_t>(mask));
    for (int64_t i = 0; i < slots; ++i) {
      const Key k = load_key(&keys[idx]);
      if (k == key) return idx;
      if (k == kEmptyKey) {
        const unsigned long long prev = atomicCAS(
            reinterpret_cast<unsigned long long*>(&keys[idx]),
            static_cast<unsigned long long>(kEmptyKey),
            static_cast<unsigned long long>(key));
        // Won the slot, or lost to a thread inserting the same key — either
        // way this slot now holds `key`.  Concurrent calls for one key must
        // agree on the row (backing contract), and they do.
        if (static_cast<Key>(prev) == kEmptyKey ||
            static_cast<Key>(prev) == key) {
          return idx;
        }
        // Lost to a different key; keep probing.
      }
      idx = (idx + 1) & mask;
    }
    return kRowAllocFailed;
  }

  __device__ void* row_ptr(Row row) const {
    return static_cast<void*>(rows + row * static_cast<Row>(elems));
  }

  __host__ __device__ int row_elems() const { return elems; }
};

// ---------------------------------------------------------------------------
// Host owner — allocates, and provides the inspection hooks tests need.
// ---------------------------------------------------------------------------

template <typename ValueT>
class StubBacking {
 public:
  using value_t = ValueT;
  using View = StubBackingView<ValueT>;

  // `slots` is rounded up to a power of two.
  StubBacking(int64_t slots, int elems) : elems_(elems) {
    if (slots <= 0) throw std::invalid_argument("StubBacking: slots must be > 0");
    if (elems <= 0) throw std::invalid_argument("StubBacking: elems must be > 0");
    slots_ = 1;
    while (slots_ < slots) slots_ <<= 1;

    check(cudaMalloc(&d_keys_, sizeof(Key) * slots_), "cudaMalloc keys");

    // Rows live in mapped pinned host memory: kernels reach them over PCIe by
    // zero-copy, exactly as a real backing would, and the host can read them
    // directly for verification without any explicit copy.
    const size_t row_bytes = sizeof(value_t) * static_cast<size_t>(slots_) *
                             static_cast<size_t>(elems_);
    check(cudaHostAlloc(&h_rows_, row_bytes, cudaHostAllocMapped),
          "cudaHostAlloc rows");
    check(cudaHostGetDevicePointer(&d_rows_, h_rows_, 0),
          "cudaHostGetDevicePointer rows");

    clear();
  }

  ~StubBacking() {
    if (d_keys_) cudaFree(d_keys_);
    if (h_rows_) cudaFreeHost(h_rows_);
  }

  StubBacking(const StubBacking&) = delete;
  StubBacking& operator=(const StubBacking&) = delete;

  View view() const { return View{d_keys_, d_rows_, slots_, elems_}; }

  int64_t slots() const { return slots_; }
  int elems() const { return elems_; }

  // -- state management ----------------------------------------------------

  void clear() {
    std::vector<Key> empty(static_cast<size_t>(slots_), kEmptyKey);
    upload_keys(empty);
    const size_t n = static_cast<size_t>(slots_) * static_cast<size_t>(elems_);
    for (size_t i = 0; i < n; ++i) h_rows_[i] = value_t{};
  }

  // Host-side insert.  Returns the row, or kRowAllocFailed when full.
  Row insert_host(Key key) {
    reject_sentinel(key);
    auto keys = download_keys();
    const Row row = insert_into(keys, key);
    if (row_valid(row)) upload_keys(keys);
    return row;
  }

  // Host-side probe.  Mirrors the device semantics exactly; tests use it as the
  // reference when checking that a device probe agreed.
  Row probe_host(Key key) const {
    const auto keys = download_keys();
    return probe_in(keys, key);
  }

  bool contains(Key key) const { return row_valid(probe_host(key)); }

  int64_t size() const {
    const auto keys = download_keys();
    int64_t n = 0;
    for (Key k : keys)
      if (k != kEmptyKey) ++n;
    return n;
  }

  // -- eviction (test fixture only) ----------------------------------------
  //
  // Drops `victims` and rebuilds the table from scratch.  Every surviving key
  // is re-inserted, so slot indices — and therefore row indices — generally
  // change.  Row payloads move with their keys.
  //
  // This is the harshest thing a backing can do to the cache, and the cache is
  // supposed to be indifferent to it: it holds keys, never rows.  Any code
  // path that cached a row index across a kernel boundary will produce wrong
  // data here and nowhere else.
  void evict_host(const std::vector<Key>& victims) {
    auto old_keys = download_keys();

    std::vector<bool> drop(static_cast<size_t>(slots_), false);
    for (Key v : victims) {
      const Row r = probe_in(old_keys, v);
      if (row_valid(r)) drop[static_cast<size_t>(r)] = true;
    }

    // Snapshot surviving (key, payload) pairs, then rebuild.
    std::vector<Key> survivors;
    std::vector<value_t> payloads;
    survivors.reserve(static_cast<size_t>(slots_));
    for (int64_t i = 0; i < slots_; ++i) {
      const size_t si = static_cast<size_t>(i);
      if (old_keys[si] == kEmptyKey || drop[si]) continue;
      survivors.push_back(old_keys[si]);
      const value_t* src = h_rows_ + i * elems_;
      payloads.insert(payloads.end(), src, src + elems_);
    }

    std::vector<Key> new_keys(static_cast<size_t>(slots_), kEmptyKey);
    std::vector<value_t> new_rows(
        static_cast<size_t>(slots_) * static_cast<size_t>(elems_), value_t{});
    for (size_t s = 0; s < survivors.size(); ++s) {
      const Row r = insert_into(new_keys, survivors[s]);
      if (!row_valid(r)) {
        throw std::runtime_error(
            "StubBacking::evict_host: rebuild failed to place a surviving key");
      }
      const value_t* src = payloads.data() + s * elems_;
      value_t* dst = new_rows.data() + static_cast<size_t>(r) * elems_;
      for (int e = 0; e < elems_; ++e) dst[e] = src[e];
    }

    upload_keys(new_keys);
    const size_t n = new_rows.size();
    for (size_t i = 0; i < n; ++i) h_rows_[i] = new_rows[i];
  }

  // -- row payload access (host side; rows are host-mapped) ----------------

  value_t* row_host(Row row) {
    return h_rows_ + static_cast<size_t>(row) * static_cast<size_t>(elems_);
  }
  const value_t* row_host(Row row) const {
    return h_rows_ + static_cast<size_t>(row) * static_cast<size_t>(elems_);
  }

  // Reads a key's row.  Returns false when the key is absent (which, after an
  // evict_host, is a legitimate outcome the caller must handle).
  bool read_row(Key key, std::vector<value_t>& out) const {
    const Row r = probe_host(key);
    if (!row_valid(r)) return false;
    const value_t* src = row_host(r);
    out.assign(src, src + elems_);
    return true;
  }

  bool write_row(Key key, const std::vector<value_t>& in) {
    if (static_cast<int>(in.size()) != elems_) {
      throw std::invalid_argument("StubBacking::write_row: wrong row width");
    }
    const Row r = probe_host(key);
    if (!row_valid(r)) return false;
    value_t* dst = row_host(r);
    for (int e = 0; e < elems_; ++e) dst[e] = in[e];
    return true;
  }

  // Whole-table snapshot as (key, payload) pairs, ordered by key.  This is what
  // the differential test compares between the cached and uncached runs; slot
  // order is deliberately not part of the comparison, since eviction and
  // rebuild are free to move rows around.
  std::vector<std::pair<Key, std::vector<value_t>>> snapshot() const {
    const auto keys = download_keys();
    std::vector<std::pair<Key, std::vector<value_t>>> out;
    for (int64_t i = 0; i < slots_; ++i) {
      const Key k = keys[static_cast<size_t>(i)];
      if (k == kEmptyKey) continue;
      const value_t* src = row_host(i);
      out.emplace_back(k, std::vector<value_t>(src, src + elems_));
    }
    std::sort(out.begin(), out.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return out;
  }

 private:
  static void reject_sentinel(Key key) {
    if (key == kEmptyKey) {
      throw std::invalid_argument(
          "StubBacking: key collides with the empty-slot sentinel");
    }
  }

  static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("StubBacking: ") + what + ": " +
                               cudaGetErrorString(e));
    }
  }

  Row probe_in(const std::vector<Key>& keys, Key key) const {
    const int64_t mask = slots_ - 1;
    int64_t idx = static_cast<int64_t>(mix64(static_cast<uint64_t>(key)) &
                                       static_cast<uint64_t>(mask));
    for (int64_t i = 0; i < slots_; ++i) {
      const Key k = keys[static_cast<size_t>(idx)];
      if (k == key) return idx;
      if (k == kEmptyKey) return kRowNotFound;
      idx = (idx + 1) & mask;
    }
    return kRowNotFound;
  }

  Row insert_into(std::vector<Key>& keys, Key key) const {
    const int64_t mask = slots_ - 1;
    int64_t idx = static_cast<int64_t>(mix64(static_cast<uint64_t>(key)) &
                                       static_cast<uint64_t>(mask));
    for (int64_t i = 0; i < slots_; ++i) {
      Key& k = keys[static_cast<size_t>(idx)];
      if (k == key) return idx;
      if (k == kEmptyKey) {
        k = key;
        return idx;
      }
      idx = (idx + 1) & mask;
    }
    return kRowAllocFailed;
  }

  std::vector<Key> download_keys() const {
    std::vector<Key> keys(static_cast<size_t>(slots_));
    check(cudaMemcpy(keys.data(), d_keys_, sizeof(Key) * slots_,
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy keys D2H");
    return keys;
  }

  void upload_keys(const std::vector<Key>& keys) {
    check(cudaMemcpy(d_keys_, keys.data(), sizeof(Key) * slots_,
                     cudaMemcpyHostToDevice),
          "cudaMemcpy keys H2D");
  }

  int64_t slots_ = 0;
  int elems_ = 0;
  Key* d_keys_ = nullptr;
  value_t* h_rows_ = nullptr;  // mapped pinned, host pointer
  value_t* d_rows_ = nullptr;  // same memory, device pointer
};

}  // namespace testing
}  // namespace lw_cache
