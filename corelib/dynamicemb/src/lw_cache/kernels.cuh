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
// Kernel bodies. One per phase, in the order ops.h lays out; the launches live
// in launch.cuh.
//
// Every kernel is a grid-stride loop, which decouples the launch shape from the
// work size and has a useful side effect: a one-thread, one-block "launch"
// still processes every element. That is what lets testing/run_host_tests.sh
// call these bodies directly under the host shim, so the only thing left
// unverified at this layer is the launch configuration itself.
//
// The row-moving kernels take the number of cooperating lanes as a template
// parameter rather than assuming 32, for the same reason.
//
// ---------------------------------------------------------------------------
// Launch order
// ---------------------------------------------------------------------------
//
//   lookup_pin
//   ---- allocate ----
//   claim            (pops the ring)
//   absorb_retry     (CLOCK only; returns spared nodes to the ring)
//   unlink           (victims leave their buckets)
//   bind             (new keys join theirs)
//   writeback        (victim rows out)
//   fill             (new rows in, then publish)
//   ---- forward, backward ----
//   release          (pushes the ring)
//
// The boundaries are not launch-granularity choices, they are correctness
// requirements; see the header comment in ops.h for what each one prevents.
//
// One cross-step constraint the launch order cannot express: a claim kernel
// must not start until every already-issued release kernel has finished, or its
// `tail` snapshot may cover a slot that a pusher reserved but has not written.
// One stream event, one direction — release never has to wait on claim. This
// holds at any prefetch depth.
//
// VERIFICATION STATUS
//   verified   — kernel bodies, via the host shim at one thread per grid
//   unverified — launch configuration, and every concurrency property the
//                shim serialises away. Needs a GPU.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "backing.h"
#include "cache.h"
#include "ops.h"

namespace lw_cache {

__host__ __device__ inline int64_t global_thread_id() {
  return static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}
__host__ __device__ inline int64_t thread_stride() {
  return static_cast<int64_t>(gridDim.x) * blockDim.x;
}

// ---------------------------------------------------------------------------
// Pin phase
// ---------------------------------------------------------------------------

// `keys` must already be deduplicated: ref counting is one +1 per unique key
// per step (INV-6), and a repeated key would add references that the single
// matching release cannot undo.
//
// Writes kNoNode for a miss. The caller compacts those into the allocate path.
template <typename ValueT>
__global__ void lookup_pin_kernel(CacheView<ValueT> view, const Key* keys,
                                  int64_t num_keys, int32_t* out_nodes) {
  for (int64_t i = global_thread_id(); i < num_keys; i += thread_stride()) {
    out_nodes[i] = cache_lookup_pin(view, keys[i]);
  }
}

// ---------------------------------------------------------------------------
// Allocate
// ---------------------------------------------------------------------------

// `slots[j].key` must be filled in by the caller. Everything else is output.
//
// `tail_snapshot` is read once on the host side (or by a preceding kernel) and
// passed in, rather than read per thread: pushes cannot happen while this
// kernel runs, so the bound is constant, and a constant costs nothing to check.
template <typename ValueT, typename BackingT>
__global__ void claim_kernel(CacheView<ValueT> view, BackingT backing,
                             AllocSlot* slots, int64_t num_slots,
                             uint64_t tail_snapshot, PopPolicy policy,
                             RetryBuffer retry, bool use_retry) {
  RetryBuffer* retry_ptr = use_retry ? &retry : nullptr;
  for (int64_t i = global_thread_id(); i < num_slots; i += thread_stride()) {
    cache_claim(view, backing, slots[i], tail_snapshot, policy, retry_ptr);
  }
}

// CLOCK only. Returns the nodes granted a second chance to the ring.
//
// Separate from claim_kernel because appending to the ring is a push, and a
// push inside a pop kernel is the overlap the whole design avoids.
//
// The counter is NOT cleared here. It cannot be: with a grid-stride loop over
// several blocks there is no point at which one thread knows the others have
// finished reading it. The launcher clears it with a stream-ordered memset
// after this returns.
//
// Leaving it uncleared is not a small bug. The count would carry over, so the
// next step re-pushes this step's entries — nodes that are already in the ring.
// Two entries for one node breaks INV-4, inflates `tail`, and hands the same
// node to two threads, which corrupts rows. It also only shows up under real
// concurrency: at low speed few second chances are granted and the count stays
// small enough to hide.
template <typename ValueT>
__global__ void absorb_retry_kernel(CacheView<ValueT> view, RetryBuffer retry) {
  // Clamp: a burst of second chances can push the reservation past the buffer,
  // and entries beyond it were never written.
  uint32_t n = *retry.count;
  if (n > static_cast<uint32_t>(retry.capacity)) {
    n = static_cast<uint32_t>(retry.capacity);
  }
  for (int64_t i = global_thread_id(); i < static_cast<int64_t>(n);
       i += thread_stride()) {
    ring_absorb_retry(view.ring, retry, static_cast<uint32_t>(i));
  }
}

template <typename ValueT, typename BackingT>
__global__ void unlink_kernel(CacheView<ValueT> view, BackingT backing,
                              AllocSlot* slots, int64_t num_slots) {
  for (int64_t i = global_thread_id(); i < num_slots; i += thread_stride()) {
    cache_unlink_victim(view, backing, slots[i]);
  }
}

template <typename ValueT>
__global__ void bind_kernel(CacheView<ValueT> view, AllocSlot* slots,
                            int64_t num_slots) {
  for (int64_t i = global_thread_id(); i < num_slots; i += thread_stride()) {
    cache_bind(view, slots[i]);
  }
}

// ---------------------------------------------------------------------------
// Row movement
// ---------------------------------------------------------------------------
//
// P0 keeps the two directions in separate kernels. Merging them — one warp
// reading the victim row into registers, issuing the load and the store
// together, then overwriting — puts both PCIe directions in flight at once, and
// in steady state their volumes are equal, since every miss causes exactly one
// eviction. That is a P1 change and it is contingent on the link actually being
// full duplex, which is a measurement, not an assumption.

template <int kLanes, typename ValueT, typename BackingT>
__global__ void writeback_kernel(CacheView<ValueT> view, BackingT backing,
                                 const AllocSlot* slots, int64_t num_slots) {
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t s = tid / kLanes; s < num_slots; s += thread_stride() / kLanes) {
    cache_writeback(view, backing, slots[s], lane, kLanes);
  }
}

// Ends by publishing: LOADING -> RESIDENT, once the row is fully written. Lane
// 0 does it, after a warp sync, because a node must not become visible while
// part of its row is still in flight.
template <int kLanes, typename ValueT, typename BackingT, typename Init>
__global__ void fill_kernel(CacheView<ValueT> view, BackingT backing,
                            const AllocSlot* slots, int64_t num_slots,
                            Init init) {
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t s = tid / kLanes; s < num_slots; s += thread_stride() / kLanes) {
    cache_fill(view, backing, slots[s], init, lane, kLanes);
    __syncwarp();
    if (lane == 0) cache_publish(view, slots[s]);
  }
}

// ---------------------------------------------------------------------------
// Slot marshalling
// ---------------------------------------------------------------------------
//
// The allocate passes work on a compacted array of misses. These move keys in
// and node ids out, so callers never have to know AllocSlot's layout.
//
// `static` rather than `inline`: nvcc ignores `inline` on a __global__ function
// (warning #20050-D) and would then emit a duplicate symbol from every
// translation unit that includes this header. Internal linkage per TU costs a
// little code size and links.

static __global__ void slots_set_keys_kernel(AllocSlot* slots, const Key* keys,
                                             int64_t n) {
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    slots[i] = AllocSlot{};
    slots[i].key = keys[i];
  }
}

// Compact the misses into a slot array, entirely on the device.
//
// The obvious alternative — pull the node array back, filter on the host, push
// the misses down — costs a synchronisation in the middle of the allocate path.
// With a prefetch depth of one that is merely wasteful; deeper than one it is
// fatal to the point of the pipeline, because the host sync orders allocate
// against release and hides whether the stream event that is *supposed* to
// order them actually does.
//
// Order within the slot array is arbitrary (atomicAdd), which is fine: nothing
// downstream depends on it, and `src` carries each slot back to its origin.
template <typename ValueT>
__global__ void compact_misses_kernel(const int32_t* nodes, const Key* keys,
                                      int64_t n, AllocSlot* slots,
                                      uint32_t* count) {
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    if (nodes[i] != kNoNode) continue;
    const uint32_t at = atomicAdd(count, 1u);
    AllocSlot s;
    s.key = keys[i];
    s.src = static_cast<int32_t>(i);
    slots[at] = s;
  }
}

// Write the allocated nodes back where their keys came from.
static __global__ void scatter_nodes_kernel(const AllocSlot* slots,
                                            const uint32_t* count,
                                            int32_t* nodes) {
  const uint32_t n = *count;
  for (int64_t i = global_thread_id(); i < static_cast<int64_t>(n);
       i += thread_stride()) {
    nodes[slots[i].src] = slots[i].node;
  }
}

// Multi-table variant: each slot also carries the table it belongs to, because
// after compaction the misses are no longer grouped by table.
static __global__ void mt_slots_set_kernel(AllocSlot* slots, const Key* keys,
                                           const int32_t* tables, int64_t n) {
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    slots[i] = AllocSlot{};
    slots[i].key = keys[i];
    slots[i].table = tables[i];
  }
}

static __global__ void slots_get_nodes_kernel(const AllocSlot* slots,
                                              int32_t* nodes, int64_t n) {
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    nodes[i] = slots[i].node;
  }
}

// ---------------------------------------------------------------------------
// Device-count variants
// ---------------------------------------------------------------------------
//
// Identical to the ones above except that the number of slots is read from
// device memory. Launched over the worst case; threads past the real count
// return immediately. This is what lets the whole allocate path run without the
// host ever learning how many misses there were, and therefore without a
// synchronisation that would mask the release-to-allocate event.

template <typename ValueT, typename BackingT>
__global__ void claim_kernel_n(CacheView<ValueT> view, BackingT backing,
                               AllocSlot* slots, const uint32_t* count,
                               uint64_t tail_snapshot, PopPolicy policy,
                               RetryBuffer retry, bool use_retry) {
  const int64_t n = static_cast<int64_t>(*count);
  RetryBuffer* retry_ptr = use_retry ? &retry : nullptr;
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    cache_claim(view, backing, slots[i], tail_snapshot, policy, retry_ptr);
  }
}

template <typename ValueT, typename BackingT>
__global__ void unlink_kernel_n(CacheView<ValueT> view, BackingT backing,
                                AllocSlot* slots, const uint32_t* count) {
  const int64_t n = static_cast<int64_t>(*count);
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    cache_unlink_victim(view, backing, slots[i]);
  }
}

template <typename ValueT>
__global__ void bind_kernel_n(CacheView<ValueT> view, AllocSlot* slots,
                              const uint32_t* count) {
  const int64_t n = static_cast<int64_t>(*count);
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    cache_bind(view, slots[i]);
  }
}

template <int kLanes, typename ValueT, typename BackingT>
__global__ void writeback_kernel_n(CacheView<ValueT> view, BackingT backing,
                                   const AllocSlot* slots,
                                   const uint32_t* count) {
  const int64_t n = static_cast<int64_t>(*count);
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t s = tid / kLanes; s < n; s += thread_stride() / kLanes) {
    cache_writeback(view, backing, slots[s], lane, kLanes);
  }
}

template <int kLanes, typename ValueT, typename BackingT, typename Init>
__global__ void fill_kernel_n(CacheView<ValueT> view, BackingT backing,
                              const AllocSlot* slots, const uint32_t* count,
                              Init init) {
  const int64_t n = static_cast<int64_t>(*count);
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t s = tid / kLanes; s < n; s += thread_stride() / kLanes) {
    cache_fill(view, backing, slots[s], init, lane, kLanes);
    __syncwarp();
    if (lane == 0) cache_publish(view, slots[s]);
  }
}

// ---------------------------------------------------------------------------
// Release
// ---------------------------------------------------------------------------

// Takes the node ids produced by lookup_pin plus allocate, one per unique key,
// exactly one release for each pin (INV-6).
template <typename ValueT>
__global__ void release_kernel(CacheView<ValueT> view, const int32_t* nodes,
                               int64_t num_nodes) {
  for (int64_t i = global_thread_id(); i < num_nodes; i += thread_stride()) {
    cache_release(view, nodes[i]);
  }
}

// ---------------------------------------------------------------------------
// Flush
// ---------------------------------------------------------------------------

// Requires a drained pipeline. Writes every RESIDENT row with no filter — see
// cache_flush_node() for why filtering, and `ref > 0` in particular, would be
// data loss.
template <int kLanes, typename ValueT, typename BackingT>
__global__ void flush_kernel(CacheView<ValueT> view, BackingT backing) {
  const int64_t tid = global_thread_id();
  const int lane = static_cast<int>(tid % kLanes);
  for (int64_t n = tid / kLanes; n < view.capacity;
       n += thread_stride() / kLanes) {
    cache_flush_node(view, backing, static_cast<int32_t>(n), lane, kLanes);
  }
}

// ---------------------------------------------------------------------------
// Consistency sweep (debug)
// ---------------------------------------------------------------------------

// Counts nodes still holding references. After a drain every one of them must
// be zero; a survivor is an INV-6 leak, which otherwise shows up only as a hit
// rate that decays over hours and has no obvious cause.
template <typename ValueT>
__global__ void count_referenced_kernel(CacheView<ValueT> view,
                                        unsigned long long* out) {
  for (int64_t n = global_thread_id(); n < view.capacity; n += thread_stride()) {
    if (ref_of(view.states[n]) != 0u) atomicAdd(out, 1ull);
  }
}

}  // namespace lw_cache
