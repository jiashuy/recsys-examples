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
// Per-key operations. Each is what one thread (or one warp, for the row moves)
// does; the __global__ wrappers live in kernels.cuh.
//
// Keeping the logic here rather than inside kernels is what lets
// testing/run_host_tests.sh exercise it without a GPU.
//
// ---------------------------------------------------------------------------
// The phase structure, and why it is not negotiable
// ---------------------------------------------------------------------------
//
//   lookup      | pin phase        | reads chains, pins hits
//   ---- kernel boundary ----
//   claim       | allocate pass 1  | resolve new row, take a node from the ring
//   ---- kernel boundary ----
//   unlink      | allocate pass 2  | victims leave their buckets
//   ---- kernel boundary ----
//   bind        | allocate pass 3  | new keys join their buckets
//   ---- kernel boundary ----
//   transfer    | allocate pass 4  | write back the victim row, load the new one
//   ---- kernel boundary ----
//   release     | end of backward  | drop the reference, maybe enqueue
//
// Every boundary is load-bearing:
//
//   lookup / claim   a node found by lookup can be claimed and rebound before
//                    the pin lands, and would then read RESIDENT again and pin
//                    successfully onto the wrong key (INV-8)
//   claim / unlink   claim pops the ring; unlink must not run while pops are in
//                    flight for the ordinary reason that it reads chains
//   unlink / bind    removal and insertion in one kernel interleave on the same
//                    bucket — the lock-free doubly-linked list problem this
//                    design sidesteps rather than solves
//   * / release      release pushes the ring; a pop kernel must not start while
//                    a push is in flight, or its `tail` snapshot may include a
//                    reserved-but-unwritten slot
//
// The last one is the only cross-phase constraint that needs a stream event,
// and it is one-directional: allocate waits for release, never the reverse.
//
// VERIFICATION STATUS
//   verified   — single-threaded behaviour of every operation, including the
//                miss / hit / bypass paths and backing eviction, via
//                testing/run_host_tests.sh
//   unverified — anything concurrent. Needs a GPU.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "backing.h"
#include "cache.h"
#include "hash_chain.h"
#include "node_state.h"
#include "ring.h"

namespace lw_cache {

// Per-miss state carried across the allocate passes.
struct AllocSlot {
  Key key = kUnboundKey;          // the incoming key
  int32_t table = 0;              // which table this key belongs to
  int32_t src = -1;               // index in the caller's key array, for scatter-back
  int32_t node = kNoNode;         // kNoNode => not cached this step (see below)
  Key victim_key = kUnboundKey;   // kUnboundKey => the node held nothing
  Row new_row = kRowNotFound;     // where the incoming key lives in the backing
  Row victim_row = kRowNotFound;  // where the outgoing key must be written
  bool fresh = false;             // backing had no record: initialize, no H2D

  // Why `node == kNoNode` after claim, and what the caller must do:
  //   backing full      — the key cannot be cached at all; route it through the
  //                       bypass path (temporary buffer + initializer)
  //   ring exhausted    — every node is pinned by an in-flight step, i.e. the
  //                       capacity planning is wrong. Not recoverable by
  //                       retrying; report it.
  __host__ __device__ bool allocated() const { return node != kNoNode; }
};

// ---------------------------------------------------------------------------
// Lookup + pin
// ---------------------------------------------------------------------------

// Resolve `key` to a pinned node, or kNoNode for a miss.
//
// Two ways a chain hit still ends up a miss, both deliberately not retried:
//
//   * try_pin failed — an allocator claimed the node first. Waiting would mean
//     spinning on an operation that contains a PCIe transfer.
//   * the key changed — the node was evicted and rebound between the walk and
//     the pin, and the state would read RESIDENT again (INV-8). Rare enough
//     that the extra 8-byte read costs nothing, wrong often enough to matter if
//     omitted.
//
// A miss here costs one row transfer, never correctness.
template <typename ValueT>
__device__ inline int32_t cache_lookup_pin(const CacheView<ValueT>& view,
                                           Key key) {
  view.metrics.bump(kMetricLookups);
  const int32_t n = chain_lookup(view.chain, key);
  if (n == kNoNode) return kNoNode;
  if (!try_pin(&view.states[n])) {
    view.metrics.bump(kMetricPinFailed);
    return kNoNode;
  }
  if (view.chain.keys[n] != key) {
    // Undo the pin. This cannot drive ref to zero: whoever rebound the node
    // holds a reference of their own.
    release_ref(&view.states[n]);
    view.metrics.bump(kMetricIdentityMismatch);
    return kNoNode;
  }
  view.metrics.bump(kMetricHits);
  return n;
}

// ---------------------------------------------------------------------------
// Allocate, pass 1 — resolve the row, then take a node
// ---------------------------------------------------------------------------

// The order matters. Taking a node first and discovering afterwards that the
// backing is full would leave a claimed node to give back, and giving it back
// means pushing the ring from inside a pop kernel — exactly the overlap the
// design rules out. Resolving the row first makes the failure free.
//
// If the ring is exhausted *after* the row was reserved, the reservation simply
// stands: the key now has a backing row and no cache node, which is the state
// any uncached key is in.
template <typename ValueT, typename BackingT>
__device__ inline void cache_claim(const CacheView<ValueT>& view,
                                   BackingT& backing, AllocSlot& slot,
                                   uint64_t tail_snapshot,
                                   const PopPolicy& policy, RetryBuffer* retry) {
  slot.node = kNoNode;
  slot.victim_key = kUnboundKey;
  slot.victim_row = kRowNotFound;

  // Was the key already known to the backing? The answer decides whether the
  // row has to come over PCIe or can be generated in place. A key the backing
  // evicted looks exactly like one it has never seen, and is treated the same:
  // its history is gone either way.
  slot.new_row = backing.probe(slot.key);
  slot.fresh = !row_valid(slot.new_row);
  view.metrics.bump(slot.fresh ? kMetricFresh : kMetricLoaded);
  if (slot.fresh) {
    slot.new_row = backing.find_or_insert(slot.key);
    if (!row_valid(slot.new_row)) {
      view.metrics.bump(kMetricBackingFull);
      return;  // bypass
    }
  }

  const PopResult pr =
      ring_pop(view.ring, view.states, tail_snapshot, policy, retry);
  if (pr.node == kNoNode) return;  // ring dry -> capacity error

  slot.node = pr.node;
  // A node that never held a row has nothing to unlink and nothing to write
  // back. Anything else does, unconditionally — there is no dirty bit, because
  // a resident row is never older than its backing copy (INV-9).
  if (needs_writeback(pr.claimed_from)) {
    slot.victim_key = view.chain.keys[pr.node];
  }
}

// ---------------------------------------------------------------------------
// Allocate, pass 2 — victims leave their buckets
// ---------------------------------------------------------------------------

// Also resolves where the victim's row must be written. Probing here rather
// than reading a stored index is what makes the cache indifferent to the
// backing moving rows around, or dropping the key entirely: if the backing has
// forgotten this key, re-insert it and write to the new row. The cached copy is
// the newest one that exists (INV-9), so nothing is lost.
template <typename ValueT, typename BackingT>
__device__ inline void cache_unlink_victim(const CacheView<ValueT>& view,
                                           BackingT& backing, AllocSlot& slot) {
  if (!slot.allocated() || slot.victim_key == kUnboundKey) return;

  chain_unlink(view.chain, slot.node, slot.victim_key);

  slot.victim_row = backing.probe(slot.victim_key);
  if (!row_valid(slot.victim_row)) {
    slot.victim_row = backing.find_or_insert(slot.victim_key);
    // Still nothing? The backing is full and this key's updates are dropped.
    // Loud in debug; in release the row transfer simply skips it.
    LW_CACHE_ASSERT(row_valid(slot.victim_row) &&
                    "lw_cache: no backing row for an evicted key; its updates "
                    "since it entered the cache are lost");
  }
}

// ---------------------------------------------------------------------------
// Allocate, pass 3 — new keys join their buckets
// ---------------------------------------------------------------------------

template <typename ValueT>
__device__ inline void cache_bind(const CacheView<ValueT>& view,
                                  AllocSlot& slot) {
  if (!slot.allocated()) return;
  view.chain.keys[slot.node] = slot.key;
  // EVICTING -> LOADING, pinned for the requesting step, recency cleared, not
  // in the ring. The row is not readable until transfer marks it RESIDENT.
  view.states[slot.node] = loading_state();
  chain_insert(view.chain, slot.node);
}

// ---------------------------------------------------------------------------
// Allocate, pass 4 — move the rows
// ---------------------------------------------------------------------------
//
// P0 keeps the two directions as separate calls. Merging them into one warp
// that reads the old row into registers, issues the load and the store
// together, and then overwrites — so both PCIe directions are in flight at once
// — is a P1 change and depends on measuring that the link really is full
// duplex.
//
// `lane` / `lanes` let one warp cooperate on a row; pass (0, 1) for a single
// thread.

template <typename ValueT, typename BackingT>
__device__ inline void cache_writeback(const CacheView<ValueT>& view,
                                       const BackingT& backing,
                                       const AllocSlot& slot, int lane,
                                       int lanes) {
  if (!slot.allocated() || !row_valid(slot.victim_row)) return;
  if (lane == 0) view.metrics.bump(kMetricWrittenBack);
  const ValueT* src = view.row(slot.node);
  ValueT* dst = static_cast<ValueT*>(backing.row_ptr(slot.victim_row));
  for (int i = lane; i < view.row_elems; i += lanes) dst[i] = src[i];
}

// `Init` is called for a key the backing has no record of. Signature:
//   ValueT operator()(Key key, int elem) const
template <typename ValueT, typename BackingT, typename Init>
__device__ inline void cache_fill(const CacheView<ValueT>& view,
                                  const BackingT& backing, const AllocSlot& slot,
                                  const Init& init, int lane, int lanes) {
  if (!slot.allocated()) return;
  ValueT* dst = view.row(slot.node);
  if (slot.fresh) {
    for (int i = lane; i < view.row_elems; i += lanes) dst[i] = init(slot.key, i);
  } else {
    const ValueT* src = static_cast<const ValueT*>(backing.row_ptr(slot.new_row));
    for (int i = lane; i < view.row_elems; i += lanes) dst[i] = src[i];
  }
}

// Publish the row. Only lane 0 of a cooperating warp should call this, after
// the row is fully written.
template <typename ValueT>
__device__ inline void cache_publish(const CacheView<ValueT>& view,
                                     const AllocSlot& slot) {
  if (!slot.allocated()) return;
  __threadfence();  // the row must be visible before the state says it is
  mark_resident(&view.states[slot.node]);
}

// ---------------------------------------------------------------------------
// Flush
// ---------------------------------------------------------------------------

// Write one node's row back, for checkpointing.
//
// Requires a drained pipeline: every in-flight step must have finished its
// backward pass. Reading a row while the optimizer is still writing it yields a
// torn row, and reading one while an allocate is overwriting it yields another
// key's data. Draining removes both, and gives a free check of INV-6 — with no
// step in flight every ref must be zero, so a non-zero one is a leak, which
// otherwise only shows up as a hit rate that sags over hours.
//
// Every RESIDENT row is written, with no condition attached. Skipping any is
// data loss, not a saving: a hot key that has stayed cached for thousands of
// steps has never been written back at all, so the backing copy is that far
// behind (INV-9). `ref > 0` in particular would be the worst filter to apply —
// it selects precisely the keys currently in use.
template <typename ValueT, typename BackingT>
__device__ inline void cache_flush_node(const CacheView<ValueT>& view,
                                        BackingT& backing, int32_t node,
                                        int lane, int lanes) {
  const StateWord st = *reinterpret_cast<volatile StateWord*>(&view.states[node]);
  LW_CACHE_ASSERT(ref_of(st) == 0 &&
                  "lw_cache: flush on a node with outstanding references; "
                  "either the pipeline was not drained or a release was missed");
  if (state_of(st) != kResident) return;

  const Key key = view.chain.keys[node];
  Row row = backing.probe(key);
  if (!row_valid(row)) row = backing.find_or_insert(key);
  if (!row_valid(row)) {
    LW_CACHE_ASSERT(false &&
                    "lw_cache: no backing row to flush into; updates dropped");
    return;
  }

  const ValueT* src = view.row(node);
  ValueT* dst = static_cast<ValueT*>(backing.row_ptr(row));
  for (int i = lane; i < view.row_elems; i += lanes) dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Release
// ---------------------------------------------------------------------------

// Drop this step's reference. Exactly one +1 per unique key per step must be
// matched by exactly one -1 (INV-6); an unbalanced release borrows out of the
// ref field and corrupts the rest of the state word, so the assertion inside
// release_ref() is worth keeping in debug builds.
template <typename ValueT>
__device__ inline void cache_release(const CacheView<ValueT>& view,
                                     int32_t node) {
  if (node == kNoNode) return;
  if (release_ref(&view.states[node])) ring_push(view.ring, node);
}

}  // namespace lw_cache
