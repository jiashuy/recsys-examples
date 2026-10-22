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

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace lw_cache {

// ---------------------------------------------------------------------------
// Backing contract
// ---------------------------------------------------------------------------
//
// The cache owns GPU-resident rows and decides *which* keys are resident.
// The backing owns the authoritative store for *all* keys and decides where a
// key's row lives.  These are the only two things the cache asks of it:
//
//   1. resolve a key to a row index
//   2. hand out a device-addressable pointer to that row
//
// The cache never stores a row index across kernel boundaries.  It resolves a
// key to a row at the moment it needs to read or write that row, and discards
// the result immediately after.  This is deliberate and buys two properties:
//
//   * immunity to backing rehash / compaction — a stale row index can never be
//     used, because none is ever kept
//   * tolerance of backing eviction — see `probe` below
//
// See docs/lightweight_cache_design.md for the full design.
//
// ---------------------------------------------------------------------------
// Required interface
// ---------------------------------------------------------------------------
//
// A backing implementation supplies a trivially-copyable *view* struct that is
// passed to cache kernels by value.  Cache kernels are templated on the view
// type, so there are no virtual calls on the device.  The view must provide:
//
//   __device__ Row probe(Key key) const;
//   __device__ Row find_or_insert(Key key);
//   __device__ void* row_ptr(Row row) const;
//   __host__ __device__ int row_elems() const;
//
// plus a nested `using value_t = ...` naming the row element type.
//
// Use LW_CACHE_STATIC_ASSERT_BACKING(T) to check a candidate at compile time.
//
// ---------------------------------------------------------------------------
// Semantics
// ---------------------------------------------------------------------------
//
// probe(key)
//   Resolve `key` without inserting.  Returns the row index, or `kRowNotFound`.
//
//   IMPORTANT: `kRowNotFound` does not mean "this key was never seen".  The
//   backing is permitted to evict, so a key the cache is currently holding may
//   have been dropped from the backing since it was loaded.  Callers must treat
//   not-found as "the backing has no record of this key" and act accordingly:
//
//     - on the miss path   -> initialize a fresh row (identical handling to a
//                             genuinely new key; backing eviction means the
//                             key's history is forgotten, which is the same
//                             thing)
//     - on the evict path  -> call find_or_insert() to obtain a new row and
//                             write there.  No data is lost: the cache copy is
//                             by construction the newest version (INV-9).
//
// find_or_insert(key)
//   Resolve `key`, inserting it if absent.  Returns the row index, or
//   `kRowAllocFailed` if the backing cannot accommodate it.  The row's contents
//   are *undefined* on insert — this call reserves a row, it does not
//   initialize one.  Callers must write the row before any subsequent read.
//
//   A failed allocation is not fatal to the cache: the caller routes the key
//   through the non-admitted bypass path.  It must never be silently ignored,
//   because doing so would leave a resident node pointing at no backing row.
//
// row_ptr(row)
//   Device-addressable pointer to the first element of `row`.  Rows normally
//   live in mapped pinned host memory and are reached over PCIe by zero-copy;
//   the cache reads and writes them directly from kernels and never issues an
//   explicit memcpy.  The pointer is valid only for rows currently allocated in
//   the backing.
//
// row_elems()
//   Number of `value_t` elements per row: embedding dims + optimizer state
//   dims.  Embedding and optimizer state share one contiguous row and one
//   dtype, so a row transfer is a single coalesced move.
//
// ---------------------------------------------------------------------------
// Concurrency
// ---------------------------------------------------------------------------
//
// The cache calls these from many threads at once.  Required guarantees:
//
//   * probe          — safe concurrently with other probe and find_or_insert
//   * find_or_insert — safe concurrently with other find_or_insert and probe;
//                      concurrent calls for the *same* key must return the
//                      *same* row (the cache deduplicates keys before calling,
//                      so this arises only across kernels, but the guarantee
//                      keeps the contract independent of that)
//   * row_ptr        — pure, always safe
//
// Not required: safety against a concurrent backing-side rehash, eviction
// sweep, or capacity change.  The cache assumes such operations are serialized
// against cache kernels by the caller (stream ordering).
//
// ---------------------------------------------------------------------------

using Key = int64_t;
using Row = int64_t;

// probe(): key has no record in the backing.
constexpr Row kRowNotFound = -1;
// find_or_insert(): backing is full and cannot admit the key.
constexpr Row kRowAllocFailed = -2;

__host__ __device__ inline bool row_valid(Row row) { return row >= 0; }

// ---------------------------------------------------------------------------
// Compile-time contract check
// ---------------------------------------------------------------------------

namespace detail {

template <typename T, typename = void>
struct has_backing_interface : std::false_type {};

template <typename T>
struct has_backing_interface<
    T,
    std::void_t<
        typename T::value_t,
        decltype(std::declval<const T&>().probe(std::declval<Key>())),
        decltype(std::declval<T&>().find_or_insert(std::declval<Key>())),
        decltype(std::declval<const T&>().row_ptr(std::declval<Row>())),
        decltype(std::declval<const T&>().row_elems())>> : std::true_type {};

}  // namespace detail

// A backing view is passed to kernels by value, so it must be trivially
// copyable — it may hold pointers and scalars, never ownership.
#define LW_CACHE_STATIC_ASSERT_BACKING(T)                                    \
  static_assert(::lw_cache::detail::has_backing_interface<T>::value,         \
                #T " does not satisfy the lw_cache backing interface: "      \
                   "expected value_t, probe(), find_or_insert(), row_ptr(), " \
                   "row_elems()");                                            \
  static_assert(std::is_trivially_copyable<T>::value,                        \
                #T " must be trivially copyable: backing views are passed "  \
                   "to kernels by value and must not own resources")

}  // namespace lw_cache
