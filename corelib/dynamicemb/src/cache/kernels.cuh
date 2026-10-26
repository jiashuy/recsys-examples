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
// rows.cuh does, and none of this needs it. That split is deliberate rather than
// incidental: placement is where all the concurrency is, and it can be run,
// raced and checked on a device with a single integer standing in for a row.
//
// ---------------------------------------------------------------------------
// Launch order
// ---------------------------------------------------------------------------
//
//   lookup_pin          which keys are already resident, and hold them
//   ---- allocate ----
//   allocate            take a node per miss, and rebind it to the new key
//   writeback, fill, publish   in rows.cuh: victim rows out, rows with a history in,
//                       new keys' rows initialised by the caller in between
//   ---- forward, backward ----
//   release             give the references back, and re-queue what went idle
//
// Around the backing's insert, which evicts keys of its own:
//
//   EvictionGuard       asked by the backing's eviction, per candidate: may it go?
//                       (this and the next two are launched over every table at once)
//   ---- backing insert ----
//   find_nodes          which of the evicted keys are cached
//   fetch_rows          in rows.cuh: takes the latest values off those nodes
//
// A checkpoint, with the pipeline drained:
//
//   resident_slots      one slot per resident node, shaped like an eviction's
//   writeback           in rows.cuh, unchanged: the copy is the same
//   invalidate          drop those bindings; the node is empty, nothing to write back
//
// Two boundaries here are correctness requirements rather than launch-
// granularity choices:
//
//   * lookup_pin must not overlap allocate. HashMap's rule 1 — a lookup runs
//     concurrently with nothing — depends on it, and so does pinning: a node
//     found by lookup that is evicted and rebound before the pin lands reads
//     RESIDENT again, and the pin would succeed onto the wrong key. Nothing in
//     the word can tell the two apart, so the separation is the whole defence.
//   * allocate must not start until every already-issued release has finished,
//     and a release issued later must not start until allocate's tail snapshot
//     has been taken (ReclaimQueueStorage::take_snapshot). A releasing kernel
//     pushes; an allocating one pops; and a pop whose bound covers a slot a
//     pusher has reserved but not yet written reads the previous occupant. Once
//     the snapshot is taken, a release may overlap the allocate kernel freely.
//     See reclaim_queue.h.
//
//   * find_nodes and invalidate are two kernels, not one. A lookup runs
//     concurrently with nothing, and invalidate removes from the bucket chains.
//     The fetch of the latest values sits between them, and has to: invalidate is
//     what lets a later allocate take the node.
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

#include "hashmap.h"
#include "node_state.h"
#include "reclaim_queue.h"

namespace cache {

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

  // What it held before, meaningful only when has_victim: a node that was empty
  // never held anything, and there is nothing to write back.
  KeyT victim_key;
  bool has_victim;
};

// ---------------------------------------------------------------------------
// The tables of a step
// ---------------------------------------------------------------------------
//
// The kernels below that take `offsets` serve every table in one launch. The keys
// come grouped by table, and `offsets` -- on the device, one entry per table and the
// total at the end -- says where each table's keys start: block row `blockIdx.y`
// is table `blockIdx.y`, and works on [offsets[t], offsets[t + 1]). So a step's
// per-table counts never have to reach the host: the launch is sized by the total
// number of keys, and a block whose table has fewer finds its range empty and leaves.
//
// `maps` and `queues` are the per-table arrays, on the device. The kernels that
// touch a reclaim queue are block-collective (see allocate_kernel), and stay so:
// every thread of a block is on the same table.

// ---------------------------------------------------------------------------
// lookup_pin
// ---------------------------------------------------------------------------
//
// For each unique key: resolve it, and hold the node if it is resident.
//
//   out_nodes[i]     the node holding keys[i], or kNoNode for a miss
//
// A pin that is refused is reported as a miss rather than waited for, because
// spinning would mean spinning on an operation that may contain a PCIe
// transfer. That is only safe because the refusal cannot happen here: a node is
// loading or claimed only inside an allocate, and every row it loads is
// published before the next lookup runs. Were it to happen, the allocator would
// bind the key to a second node while the first still holds it.
//
// Ordinary grid-stride, nothing here is block-collective. This kernel chases
// pointers through bucket chains, and it depends on a block's warps running out
// of step with each other to hide the latency.

template <typename MapT>
__global__ void lookup_pin_kernel(const MapT* maps, const int64_t* offsets,
                                  const typename MapT::Key* keys,
                                  typename MapT::NodeId* out_nodes) {
  using NodeId = typename MapT::NodeId;
  const MapT map = maps[blockIdx.y];
  const int64_t begin = offsets[blockIdx.y];
  const int64_t n = offsets[blockIdx.y + 1] - begin;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    NodeId result = MapT::kNoNode;

    // The node found is the node pinned: rebinding lives in the allocate
    // kernel, which does not overlap this one. See the launch order above.
    const NodeId node = map.lookup(keys[begin + i]);
    if (node != MapT::kNoNode && map.nodes[node].state.try_pin()) result = node;
    out_nodes[begin + i] = result;
  }
}

// ---------------------------------------------------------------------------
// allocate
// ---------------------------------------------------------------------------
//
// For each key to bind: take a node off the reclaim queue and rebind it to the key.
//
// Claiming and rebinding are one kernel because nothing separates them. The
// node is exclusively this thread's from the moment try_claim succeeds -- it is
// LOADING with a reference held, so no pin can take it and no other claim can
// either -- and the bucket lock handles the chain. Keeping them together also
// keeps the victim's state in a register instead of in a buffer.
//
// A node that cannot be had (the queue ran dry) yields kNoNode in the slot. It
// means every node is held by an in-flight step, which is a capacity planning
// error; the caller has to route the key past the cache rather than retry.
//
// BLOCK-COLLECTIVE: see the header comment. Every thread of a block runs every
// iteration of the outer loop, and a block is on one table.

template <typename QueueT, typename MapT>
__global__ void allocate_kernel(
    const QueueT* queues, const MapT* maps, const int64_t* offsets,
    const typename MapT::Key* keys, const int64_t* positions,
    AllocSlot<typename MapT::Key, typename MapT::NodeId>* out_slots,
    typename MapT::NodeId* out_nodes) {
  using NodeId = typename MapT::NodeId;
  using Key = typename MapT::Key;
  using Slot = AllocSlot<Key, NodeId>;

  __shared__ typename QueueT::Scratch scratch;

  QueueT queue = queues[blockIdx.y];
  MapT map = maps[blockIdx.y];
  const int64_t begin = offsets[blockIdx.y];
  const int64_t n = offsets[blockIdx.y + 1] - begin;

  // One value for the whole kernel, fixed before it launched. `tail` itself may
  // be moving under a release that started later; that release writes above this
  // bound, and nothing here reads above it.
  const uint64_t tail_snapshot = queue.snapshot_tail();

  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    const bool want = i < n;

    NodeState victim = NodeState::make(NodeState::kEmpty, 0u, false, false);
    const NodeId node =
        queue.pop(want, tail_snapshot, scratch, [&](NodeId candidate) {
          return map.nodes[candidate].state.try_claim(&victim);
        });

    if (!want) continue;

    Slot slot;
    slot.node = node;
    slot.key = keys[begin + i];
    slot.victim_key = Key{};
    slot.has_victim = false;

    if (node != QueueT::kNoNode) {
      // Read the outgoing key before insert() overwrites it. Only a resident
      // victim has one worth carrying; an empty node has never held anything.
      if (victim.residency() == NodeState::kResident) {
        slot.victim_key = map.nodes[node].key;
        slot.has_victim = true;
        map.remove(node);
      }
      // Bound, and not yet readable: try_claim left the node LOADING with the
      // reference this step holds. The row lands later, and publish() is what
      // advertises it, with a release store for exactly that reason.
      map.insert(node, slot.key);
    }
    out_slots[begin + i] = slot;

    // Put the answer back where the key came from, so the caller has one dense
    // array indexed by unique key and does not need a scattering kernel. A key
    // that got no node keeps the kNoNode lookup_pin already wrote.
    out_nodes[positions[begin + i]] = slot.node;
  }
}

// ---------------------------------------------------------------------------
// Evicting by key
// ---------------------------------------------------------------------------
//
// The backing decides which keys leave the system, by its own score. The cache
// holds no score and has no say in that, except one: a key a step is using must
// not go. Both halves live here.
//
// These three are the only kernels in the cache launched over every table at
// once. The backing's eviction is one kernel over all its tables, and what it
// hands back is a list of keys from any of them in no particular order, so there
// is no per-table slice to launch over. Each takes an array of per-table maps and
// a table id per key, and picks the map per element. None of them is
// block-collective, which is what keeps that cheap; the kernels that are — the
// ones that touch the reclaim queue — are launched per table instead.
//
// EvictionGuard is the question, asked by the backing's eviction for each
// candidate it is about to take. A key that is not cached may go. A key that is
// cached may go only if its node has no references — an idle node is a cache hit
// waiting to happen, nothing more, and its row can be read for the latest values
// without anyone writing it. A referenced node is held by an in-flight step, or
// has just been taken for this one, and the answer is no.
//
// It reads the cache's chains, which is a lookup, so it may be asked only while
// the cache is not being modified. That holds: the backing inserts after the
// cache has bound this step's nodes and before it does anything else.
//
// guard.h has the same rule as a plain struct the backing's kernels can include;
// this one is for code in this directory that is generic over the map type.

template <typename MapT>
struct EvictionGuard {
  const MapT* tables;  // [num_tables], on the device

  [[nodiscard]] __forceinline__ __device__ bool may_evict(
      typename MapT::Key key, int64_t table) const {
    const MapT& map = tables[table];
    const typename MapT::NodeId node = map.lookup(key);
    return node == MapT::kNoNode || !map.nodes[node].state.load().has_refs();
  }
};

// For each key, the node that holds it in its table, or kNoNode. Pins nothing and
// changes nothing: it is the lookup half of evicting, kept apart from the removal
// because a lookup cannot overlap one.
//
// The keys are the backing's evicted keys, each evicted once. A cached one is
// idle, which the guard promised; invalidate asserts it.

template <typename MapT>
__global__ void find_nodes_kernel(const MapT* tables,
                                  const typename MapT::Key* keys,
                                  const int64_t* table_ids, int64_t n,
                                  typename MapT::NodeId* out_nodes) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    out_nodes[i] = tables[table_ids[i]].lookup(keys[i]);
  }
}

// Drop the bindings find_nodes found. Each node leaves its bucket and becomes
// EMPTY; nothing is written back, because the key is gone.
//
// Removals overlap freely — the bucket lock serialises the ones that share a
// bucket, and tables share nothing — and a node appears at most once in its
// table, because a key does. kNoNode entries, the evicted keys that were never
// cached, are skipped.

template <typename MapT>
__global__ void invalidate_kernel(const MapT* tables,
                                  const typename MapT::NodeId* node_ids,
                                  const int64_t* table_ids, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const typename MapT::NodeId node = node_ids[i];
    if (node == MapT::kNoNode) continue;
    MapT map = tables[table_ids[i]];
    map.remove(node);
    map.nodes[node].state.invalidate();
  }
}

// ---------------------------------------------------------------------------
// Flush
// ---------------------------------------------------------------------------
//
// Writing every resident row back, for a checkpoint or any time the backing has to
// hold the latest of everything. The copy is writeback_kernel's and is not
// repeated: this builds what that takes for an eviction, one slot per resident
// node with the node's own key as the victim, and the caller looks those keys up
// in the backing for their row numbers exactly as it does for real victims.
//
// Every resident node is written, whatever its references. A hot key that never
// leaves the cache is the one whose backing copy is furthest behind, and a filter
// on references would skip precisely it. But none should hold one: a flush runs
// with the pipeline drained, so every reference has been released, and
// `referenced` counts the nodes that still hold one. Zero is a free exact check of
// the whole reference protocol; anything else is a leak, which otherwise shows up
// only as a hit rate that sags over hours.
//
// Both counters must be zero on entry. Not block-collective: warp_reserve, like
// lookup_pin's compaction.

template <typename MapT, typename NodeT>
__global__ void resident_slots_kernel(
    const NodeT* nodes, int64_t capacity,
    AllocSlot<typename MapT::Key, typename MapT::NodeId>* out,
    unsigned long long* count, unsigned long long* referenced) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < capacity;
       i += int64_t(gridDim.x) * blockDim.x) {
    const NodeState state = nodes[i].state.load();
    if (state.has_refs()) atomicAdd(referenced, 1ull);

    const bool resident = state.residency() == NodeState::kResident;
    const unsigned long long j = warp_reserve(resident, count);
    if (resident) {
      AllocSlot<typename MapT::Key, typename MapT::NodeId> slot;
      slot.node = static_cast<typename MapT::NodeId>(i);
      slot.key = nodes[i].key;
      slot.victim_key = slot.key;
      slot.has_victim = true;
      out[j] = slot;
    }
  }
}

// ---------------------------------------------------------------------------
// Setting up, and reading slots back
// ---------------------------------------------------------------------------

// A fresh node pool: every node EMPTY and already in the queue, which is what the
// queue's arange(capacity) says. HashMapStorage cannot do this — it does not know
// what a state word should hold — so it is the owner's, which is this.
template <typename NodeT>
__global__ void init_nodes_kernel(NodeT* nodes, int64_t n) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    nodes[i].state.set(NodeState::make(NodeState::kEmpty, 0u, /*in_ring=*/true,
                                       /*used=*/false));
  }
}

// LOADING -> RESIDENT for a list of nodes, after their rows are written. The same
// as rows.cuh's publish_kernel, for a caller that holds node ids rather than the
// slots allocate produced.
template <typename MapT>
__global__ void publish_nodes_kernel(const MapT* maps, const int64_t* offsets,
                                     const typename MapT::NodeId* node_ids) {
  const MapT map = maps[blockIdx.y];
  const int64_t begin = offsets[blockIdx.y];
  const int64_t n = offsets[blockIdx.y + 1] - begin;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const typename MapT::NodeId node = node_ids[begin + i];
    if (node != MapT::kNoNode) map.nodes[node].state.publish();
  }
}

// AllocSlot array -> plain arrays, for a caller that cannot read the struct: the
// node each slot got, the key it evicted, and whether there was one.
template <typename MapT>
__global__ void unpack_slots_kernel(
    const AllocSlot<typename MapT::Key, typename MapT::NodeId>* slots, int64_t n,
    typename MapT::Key* victim_keys, uint8_t* has_victim,
    typename MapT::NodeId* nodes) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    victim_keys[i] = slots[i].victim_key;
    has_victim[i] = slots[i].has_victim ? 1 : 0;
    nodes[i] = slots[i].node;
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

template <typename QueueT, typename MapT>
__global__ void release_kernel(const QueueT* queues, const MapT* maps,
                               const int64_t* offsets,
                               const typename QueueT::NodeId* node_ids) {
  using NodeId = typename QueueT::NodeId;

  __shared__ typename QueueT::Scratch scratch;

  QueueT queue = queues[blockIdx.y];
  MapT map = maps[blockIdx.y];
  const int64_t begin = offsets[blockIdx.y];
  const int64_t n = offsets[blockIdx.y + 1] - begin;

  const int64_t stride = int64_t(gridDim.x) * blockDim.x;
  for (int64_t base = int64_t(blockIdx.x) * blockDim.x; base < n;
       base += stride) {
    const int64_t i = base + threadIdx.x;
    // A miss that was never admitted carries kNoNode and holds no reference.
    const NodeId node = (i < n) ? node_ids[begin + i] : QueueT::kNoNode;
    const bool owns_enqueue =
        (node != QueueT::kNoNode) && map.nodes[node].state.release();
    queue.push(owns_enqueue, node, scratch);
  }
}

}  // namespace cache
