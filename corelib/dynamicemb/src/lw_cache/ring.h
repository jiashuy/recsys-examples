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
// Reclaim ring — the queue of nodes that may be taken for reuse.
//
// Initialised to arange(C), which collapses two structures into one: there is
// no separate free list of never-used slots and eviction queue of used ones.
// Everything in the ring is simply "a node nobody is referencing".  Some have
// never held data (EMPTY), some hold a valid row whose last user has finished
// (RESIDENT, and still a cache hit until the moment it is popped).  The
// allocator does not care which.
//
// ---------------------------------------------------------------------------
// The constraint that makes this simple
// ---------------------------------------------------------------------------
//
// At any moment the ring is touched by pop-only kernels or push-only kernels,
// never both.  Stream ordering enforces it.
//
// Without that, this is the textbook MPMC ring hazard: a pusher bumps `tail` to
// reserve slot p, and before it stores the node id a popper reads `tail`, sees
// p as available, and consumes the previous occupant of that slot.  Fixing it
// properly means a per-slot sequence number (Vyukov), which is a large step up
// in complexity and in how hard it is to debug when wrong.
//
// The hazard is one-directional, which is why one event dependency suffices:
//
//   pop running, push starts   — safe.  The pusher writes at p >= the popper's
//                                tail snapshot; the popper only reads below it.
//   push in flight, pop starts — unsafe.  Exactly the window above.
//
// So: an allocate kernel must wait for every already-issued release kernel.
// The reverse is free.  This holds at any prefetch depth; deeper pipelines only
// mean a released node becomes available later, which is a capacity question,
// not a correctness one.
//
// Because pushes cannot begin mid-pop, `tail` is constant for the duration of a
// pop kernel: read it once into `tail_snapshot` and the bound check is exact
// and free for the rest of the kernel.
//
// VERIFICATION STATUS
//   verified   — pop/push protocol, stale-entry handling, exhaustion, CLOCK
//                second chances, wraparound, via testing/run_host_tests.sh
//   unverified — concurrent pop/push behaviour.  Needs a GPU.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "metrics.h"
#include "node_state.h"

namespace lw_cache {

constexpr int32_t kNoNode = -1;

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

struct RingView {
  int32_t* slots;    // [capacity], node ids
  uint64_t* head;    // monotonic; index = head & mask
  uint64_t* tail;    // monotonic
  int64_t capacity;  // power of two
  int64_t mask;      // capacity - 1
  MetricsView metrics;

  __host__ __device__ int64_t size_hint(uint64_t h, uint64_t t) const {
    return static_cast<int64_t>(t - h);
  }
};

// Landing pad for nodes granted a CLOCK second chance.  They cannot go
// straight back onto the ring: writing `tail` from inside a pop kernel is
// exactly the push/pop overlap the design rules out.  A follow-up kernel
// appends this buffer to the ring instead.
struct RetryBuffer {
  int32_t* nodes;
  uint32_t* count;
  int32_t capacity;
};

struct PopPolicy {
  bool clock;               // false => plain FIFO
  int max_second_chances;   // per calling thread; see below

  __host__ __device__ static PopPolicy fifo() { return PopPolicy{false, 0}; }
  __host__ __device__ static PopPolicy clock_with(int k) {
    return PopPolicy{true, k};
  }
};

struct PopResult {
  int32_t node;            // kNoNode when the ring ran dry
  StateWord claimed_from;  // state at the instant of claim; feeds needs_writeback()
};

// ---------------------------------------------------------------------------
// Push
// ---------------------------------------------------------------------------

// Enqueue a node whose last reference just went away.  Call only when
// release_ref() returned true — it is that call, not this one, that decides
// whether this thread owns the enqueue.
//
// No overflow check is needed and none is possible to need: in_ring admits at
// most one entry per node, so live entries never exceed capacity (INV-4).
__device__ inline void ring_push(const RingView& ring, int32_t node) {
  const uint64_t pos = atomicAdd(reinterpret_cast<unsigned long long*>(ring.tail),
                                 1ULL);
  ring.slots[pos & static_cast<uint64_t>(ring.mask)] = node;
}

// Read the bound for a pop kernel.  Valid for the whole kernel; see the header
// comment for why.
__device__ inline uint64_t ring_snapshot_tail(const RingView& ring) {
  return *reinterpret_cast<const volatile uint64_t*>(ring.tail);
}

// ---------------------------------------------------------------------------
// Pop
// ---------------------------------------------------------------------------

// Take a node for reuse, or report the ring exhausted.
//
// Three ways an entry gets skipped, all of which drop it rather than requeue
// it:
//
//   * ref != 0 — the node was hit again after being enqueued, so this entry is
//     stale.  Requeueing instead would livelock once every node is pinned:
//     threads would circle the ring forever finding nothing takeable.  Dropping
//     shrinks it monotonically until either a node is claimed or the ring runs
//     dry and the caller gets a definite failure.  Nothing leaks — in_ring is
//     cleared, so the node re-enqueues on its next release.
//
//   * try_claim lost — someone pinned it between the check and the CAS.  Same
//     treatment; the distinction does not matter.
//
//   * CLOCK second chance — see below.  This one is *not* a drop: the node
//     moves to the retry buffer and keeps in_ring set, since it still has
//     exactly one pending entry.
//
// Exhaustion means every node is pinned by an in-flight step, i.e. the capacity
// planning is wrong. It is reported, never silently worked around.
__device__ inline PopResult ring_pop(const RingView& ring, StateWord* states,
                                     uint64_t tail_snapshot,
                                     const PopPolicy& policy,
                                     RetryBuffer* retry) {
  int chances_left = policy.clock ? policy.max_second_chances : 0;

  for (;;) {
    ring.metrics.bump(kMetricPopAttempts);
    const uint64_t pos = atomicAdd(
        reinterpret_cast<unsigned long long*>(ring.head), 1ULL);
    if (pos >= tail_snapshot) {
      // Hand the slot back. Letting head run past tail would underflow the
      // unsigned difference into a huge value and every subsequent bound check
      // would pass (INV-5). Only reachable when the cache is full, which is
      // precisely when tests are least likely to be looking.
      //
      // Added rather than subtracted: CUDA has no 64-bit atomicSub, only 32-bit.
      // Adding the two's complement of one wraps to the same result.
      atomicAdd(reinterpret_cast<unsigned long long*>(ring.head), ~0ULL);
      ring.metrics.bump(kMetricRingExhausted);
      return PopResult{kNoNode, 0u};
    }

    const int32_t cand = ring.slots[pos & static_cast<uint64_t>(ring.mask)];
    StateWord st = *reinterpret_cast<volatile StateWord*>(&states[cand]);

    if (ref_of(st) != 0u) {
      clear_in_ring(&states[cand]);
      ring.metrics.bump(kMetricStaleDropped);
      continue;
    }

    // Past this point the entry was takeable. Counting the recency bit here,
    // whatever the policy, is what makes the CLOCK-versus-FIFO decision fall
    // out of an ordinary FIFO run.
    ring.metrics.bump(kMetricEvictCandidates);
    if (used_of(st)) ring.metrics.bump(kMetricUsedAtPop);

    if (policy.clock && chances_left > 0 && used_of(st)) {
      // Earned another lap: it was hit since the hand last passed. Clear the
      // bit and park it for the follow-up kernel to re-append. in_ring stays
      // set — the node still has one pending entry, just not in `slots` yet.
      if (retry != nullptr) {
        take_used(&states[cand]);
        const uint32_t at = atomicAdd(retry->count, 1u);
        if (at < static_cast<uint32_t>(retry->capacity)) {
          retry->nodes[at] = cand;
          ring.metrics.bump(kMetricSecondChances);
        } else {
          // Buffer full: undo the reservation and fall through to claiming this
          // node outright. Losing a second chance costs hit rate, never
          // correctness; losing the node would cost both.
          atomicSub(retry->count, 1u);
          clear_in_ring(&states[cand]);
          StateWord claimed_from = 0u;
          if (try_claim(&states[cand], &claimed_from)) {
            return PopResult{cand, claimed_from};
          }
          continue;
        }
      }
      --chances_left;
      continue;
    }

    // Either FIFO, or CLOCK that has run out of chances for this thread. The
    // budget guarantees termination: entries granted a second chance land in
    // the retry buffer and are invisible until the follow-up kernel, so an
    // unbounded policy could sweep every live entry into it and then report the
    // ring dry while plenty of reclaimable nodes existed. Degrading to FIFO for
    // one step is the cheap way out, and it self-corrects: the used bits just
    // cleared mean the next step behaves like CLOCK again.
    clear_in_ring(&states[cand]);
    StateWord claimed_from = 0u;
    if (try_claim(&states[cand], &claimed_from)) {
      return PopResult{cand, claimed_from};
    }
    ring.metrics.bump(kMetricClaimLost);
  }
}

// Append the CLOCK retry buffer back onto the ring. Runs as its own kernel
// after the pop kernel, keeping pushes and pops in disjoint phases.
__device__ inline void ring_absorb_retry(const RingView& ring,
                                         const RetryBuffer& retry,
                                         uint32_t i) {
  ring_push(ring, retry.nodes[i]);
}

}  // namespace lw_cache
