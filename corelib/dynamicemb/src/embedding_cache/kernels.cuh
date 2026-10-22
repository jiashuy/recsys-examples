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
// The bookkeeping kernels: which key lives in which node.
//
// Everything here decides *placement*. Nothing here moves an embedding row —
// the row pool and the backing store do not exist yet, and none of this needs
// them. That split is deliberate rather than incidental: placement is where all
// the concurrency is, and it can be run, raced and checked on a device with a
// single integer standing in for a row.
//
// ---------------------------------------------------------------------------
// Launch order
// ---------------------------------------------------------------------------
//
//   lookup_pin          which keys are already resident, and hold them
//   ---- allocate ----
//   allocate            take a node per miss, and rebind it to the new key
//   [writeback]         not yet: needs the backing
//   [fill, publish]     not yet: needs the row pool
//   ---- forward, backward ----
//   release             give the references back, and re-queue what went idle
//
// Two boundaries here are correctness requirements rather than launch-
// granularity choices:
//
//   * lookup_pin must not overlap allocate. HashMap's rule 2 — a lookup runs
//     concurrently with nothing — and the identity race below both depend on
//     it.
//   * allocate must not start until every already-issued release has finished.
//     A releasing kernel pushes; an allocating one pops; and a pop whose tail
//     snapshot covers a slot a pusher has reserved but not yet written reads
//     the previous occupant. One stream event, one direction — the reverse is
//     free. See reclaim_queue.h.
//
// What is NOT a requirement any more: splitting the rebind into an unlinking
// pass and a linking pass. That existed because removal and insertion racing on
// one bucket is the lock-free doubly-linked-list problem; the bucket lock
// settles it, and every node either operation touches is in the bucket it
// locks, because a chain never spans buckets.
//
// ---------------------------------------------------------------------------
// Block-collective launch shape
// ---------------------------------------------------------------------------
//
// allocate and release call ReclaimQueue::pop and ::push, which contain
// __syncthreads(). Every thread of a block has to reach them the same number of
// times, so those kernels stride by whole blocks and handle the tail of the
// range with a `want` flag rather than by returning early. A grid-stride loop
// written the usual way — `i = tid; i < n; i += stride` — would hang the moment
// the last wave was partial.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "ec_assert.h"
#include "hashmap.h"
#include "node_state.h"
#include "reclaim_queue.h"

namespace embedding_cache {

// ---------------------------------------------------------------------------
// What allocate hands to the row-moving kernels
// ---------------------------------------------------------------------------
//
// One per miss. The node has already been rebound by the time this is read, so
// the outgoing key has to be carried here — it is the only thing that can tell
// the writeback where the victim's row belongs in the backing.

template <typename KeyT, typename NodeIdT>
struct AllocSlot {
  NodeIdT node;  // kNoNode when the queue ran dry and the key was not admitted
  KeyT key;      // what the node holds now

  // What it held before. Meaningful only when victim_state.residency() is
  // kResident: a node that was kEmpty never held anything, and there is
  // nothing to write back. Reading the residency rather than a separate flag
  // keeps one fact in one place.
  KeyT victim_key;
  NodeState victim_state;
};

// ---------------------------------------------------------------------------
// lookup_pin
// ---------------------------------------------------------------------------
//
// For each unique key: resolve it, hold the node if it is resident, and record
// the misses densely on the way out.
//
//   out_nodes[i]     the node holding keys[i], or kNoNode for a miss
//   miss_keys[j]     the j-th missing key
//   miss_src[j]      where it came from, so allocate can write its answer back
//   *miss_count      how many there were
//
// `*miss_count` must be zero on entry — one cudaMemsetAsync on the same stream.
// It is the only thing the next kernel needs in order to size its work, and it
// stays on the device: reading it on the host would mean a synchronise in the
// middle of the prefetch pipeline, which is the one thing the whole design is
// arranged to avoid.
//
// A miss is not necessarily an absent key. try_pin refuses a node that is
// loading or already claimed, and this treats that as a miss rather than
// waiting — spinning would mean spinning on an operation that may contain a
// PCIe transfer, and the cost of being wrong is one extra transfer.
//
// Compaction here rather than in a kernel of its own: the answer is already in
// a register, so it costs one aggregated atomic and two stores instead of a
// launch and a second pass over every key.
//
// Ordinary grid-stride, and warp_reserve rather than BlockReserve, so nothing
// here is block-collective. That is deliberate. This kernel chases pointers
// through bucket chains, and it depends on a block's warps running out of step
// with each other to hide the latency; three __syncthreads() per iteration
// would line them all up again and can easily cost more than the atomics they
// save. See reserve.h.

template <typename MapT, typename NodeT>
__global__ void lookup_pin_kernel(MapT map, NodeT* nodes,
                                  const typename MapT::Key* keys, int64_t n,
                                  typename MapT::NodeId* out_nodes,
                                  typename MapT::Key* miss_keys,
                                  int64_t* miss_src,
                                  unsigned long long* miss_count) {
  using NodeId = typename MapT::NodeId;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const typename MapT::Key key = keys[i];
    NodeId result = MapT::kNoNode;

    const NodeId node = map.lookup(key);
    if (node != MapT::kNoNode && nodes[node].state.try_pin()) {
      // The node was resolved before it was held, so in principle it could have
      // been evicted and rebound in between — and the state word cannot tell,
      // because it returns to bit-identical values. Re-reading the key is what
      // settles identity.
      //
      // Under the launch order above this cannot happen: rebinding lives in the
      // allocate kernel, and that does not overlap this one. The check is here
      // for the two futures that would revive it — a fused lookup-and-allocate
      // kernel, or more than one prefetch stream — and costs one read of a line
      // the chain walk has already touched.
      if (nodes[node].key == key) {
        result = node;
      } else {
        // Hand the reference straight back. This can never be the release that
        // owns a re-queue: the node was reachable and resident, so either
        // somebody else holds it too (the count does not reach zero) or it is
        // sitting in the queue already (in_queue is set).
        const bool owns_enqueue = nodes[node].state.release();
        EC_ASSERT(!owns_enqueue &&
                  "lookup_pin: undoing a pin should never make a node newly "
                  "reclaimable");
        (void)owns_enqueue;
      }
    }
    out_nodes[i] = result;

    const bool is_miss = (result == MapT::kNoNode);
    const unsigned long long j = warp_reserve(is_miss, miss_count);
    if (is_miss) {
      miss_keys[j] = key;
      miss_src[j] = i;
    }
  }
}

// ---------------------------------------------------------------------------
// allocate
// ---------------------------------------------------------------------------
//
// For each miss: take a node off the reclaim queue and rebind it to the key.
//
// Claiming and rebinding are one kernel because nothing separates them. The
// node is exclusively this thread's from the moment try_claim succeeds — the
// residency is kEvicting, so no pin can take it and no other claim can either —
// and the bucket lock handles the chain. Keeping them together also keeps the
// victim's state in a register instead of in a buffer.
//
// A node that cannot be had (the queue ran dry) yields kNoNode in the slot. It
// means every node is held by an in-flight step, which is a capacity planning
// error; the caller has to route the key past the cache rather than retry.
//
// ---------------------------------------------------------------------------
// How this is sized
// ---------------------------------------------------------------------------
//
// The miss count lives on the device and stays there. This kernel reads it
// itself and uses it as its loop bound, so the launch can be made before it is
// known — grid_for(num_unique_keys), which grid_for caps at one wave anyway.
//
// The waste is therefore bounded by one wave of blocks that find their whole
// range past the end, and those exit before reaching a barrier and in parallel
// with each other. It does not grow with how wrong the guess was.
//
// The alternatives are all worse here. Copying the count to the host to size
// the launch is a synchronise in the middle of the prefetch pipeline, which is
// what device-side compaction exists to avoid. Dynamic parallelism replaces a
// 2-5us host launch with a 20-30us device one and forces -rdc=true on the whole
// build. A graph conditional node can skip the launch when the count is zero
// but cannot size it, and a zero count means a perfect hit rate.
//
// BLOCK-COLLECTIVE: see the header comment. Every thread of a block runs every
// iteration of the outer loop.

template <typename QueueT, typename MapT, typename NodeT>
__global__ void allocate_kernel(
    QueueT queue, MapT map, NodeT* nodes, const typename MapT::Key* miss_keys,
    const int64_t* miss_src, const unsigned long long* miss_count,
    AllocSlot<typename MapT::Key, typename MapT::NodeId>* out_slots,
    typename MapT::NodeId* out_nodes) {
  using NodeId = typename MapT::NodeId;
  using Key = typename MapT::Key;
  using Slot = AllocSlot<Key, NodeId>;

  __shared__ typename QueueT::Scratch scratch;

  // Written by lookup_pin, and visible here because a kernel boundary is a
  // full flush. Read once; nothing in this kernel changes it.
  const int64_t n = static_cast<int64_t>(*miss_count);

  // One snapshot for the whole kernel. `tail` cannot move while this runs: the
  // only thing that advances it is a pushing kernel, and the stream event above
  // keeps those from overlapping.
  const uint64_t tail_snapshot = queue.snapshot_tail();

  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    const bool want = i < n;

    NodeState victim = NodeState::make(NodeState::kEmpty, 0u, false, false);
    const NodeId node =
        queue.pop(want, tail_snapshot, scratch, [&](NodeId candidate) {
          return nodes[candidate].state.try_claim(&victim);
        });

    if (!want) continue;

    Slot slot;
    slot.node = node;
    slot.key = miss_keys[i];
    slot.victim_state = victim;
    slot.victim_key = Key{};

    if (node != QueueT::kNoNode) {
      // Read the outgoing key before insert() overwrites it. Only a resident
      // victim has one worth carrying; an empty node has never held anything.
      if (victim.residency() == NodeState::kResident) {
        slot.victim_key = nodes[node].key;
        map.remove(node);
      }
      map.insert(node, slot.key);
      // Bound, and not yet readable. The row lands later; publish() is what
      // advertises it, and it is a release store for exactly that reason.
      nodes[node].state.set(
          victim.with_residency(NodeState::kLoading).with_refs(1u).with_used(
              false).with_in_ring(false));
    }
    out_slots[i] = slot;

    // Put the answer back where the key came from, so the caller has one dense
    // array indexed by unique key and does not need a scattering kernel. A key
    // that got no node keeps the kNoNode lookup_pin already wrote.
    out_nodes[miss_src[i]] = slot.node;
  }
}

// ---------------------------------------------------------------------------
// release
// ---------------------------------------------------------------------------
//
// Give back the references a step took, and re-queue whatever went idle.
//
// Whether this thread owns the enqueue is decided inside the same atomic that
// dropped the count to zero, so there is no window in which another stream
// could take the node and this thread would advertise it as reclaimable
// anyway.
//
// BLOCK-COLLECTIVE: see the header comment.

template <typename QueueT, typename NodeT>
__global__ void release_kernel(QueueT queue, NodeT* nodes,
                               const typename QueueT::NodeId* node_ids,
                               int64_t n) {
  using NodeId = typename QueueT::NodeId;

  __shared__ typename QueueT::Scratch scratch;

  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    // A miss that was never admitted carries kNoNode and holds no reference.
    const NodeId node = (i < n) ? node_ids[i] : QueueT::kNoNode;
    const bool owns_enqueue =
        (node != QueueT::kNoNode) && nodes[node].state.release();
    queue.push(owns_enqueue, node, scratch);
  }
}

}  // namespace embedding_cache
