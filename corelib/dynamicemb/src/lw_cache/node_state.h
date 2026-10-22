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
// Node state word — the whole of the cache's concurrency control.
//
//   31 30 | 29       | 28      | 27   | 26 ................ 0
//   state | reserved | in_ring | used | ref_counter
//
// Everything lives in one 32-bit word on purpose.  Pinning a node (ref+1 on a
// hit) and claiming it for eviction (asserting ref==0 and changing state) must
// be mutually exclusive.  Split across two words there is a window where a
// thread has just incremented ref while another has already decided the node is
// free; packed into one word, both are CAS on the same address and the hardware
// settles it.
//
// A consequence worth stating loudly: `release()` decrements the *whole word*.
// That is only equivalent to "decrement ref" while ref > 0.  An unbalanced
// release borrows out of the ref field and corrupts used / in_ring / state all
// at once, turning a bookkeeping slip into a node in an impossible state.  This
// is why INV-6 (exactly one +1 and one -1 per unique key per step) is checked
// with an assertion rather than trusted.
//
// VERIFICATION STATUS
//   verified   — bit layout, accessors, and every transition's single-threaded
//                behaviour, via testing/run_host_tests.sh
//   unverified — behaviour under real contention.  The host shim serialises
//                every atomic, so it can show a protocol is right and cannot
//                show it is race-free.  Needs a GPU.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "backing.h"

#ifndef LW_CACHE_ASSERT
#ifdef NDEBUG
#define LW_CACHE_ASSERT(cond) ((void)0)
#else
#include <cassert>
#define LW_CACHE_ASSERT(cond) assert(cond)
#endif
#endif

namespace lw_cache {

using StateWord = uint32_t;

enum NodeState : uint32_t {
  // Never held data.  Claimable; must not be written back on eviction.
  kEmpty = 0u,
  // Bound to a key, contents not yet valid (H2D or initializer in flight).
  // Neither consumable nor claimable.
  kLoading = 1u,
  // Contents valid.  ref > 0 means in use by an in-flight step; ref == 0 means
  // idle but still a cache hit — the node keeps serving until it is popped.
  kResident = 2u,
  // Claimed by an allocating thread.  Pins must fail.
  kEvicting = 3u,
};

constexpr int kStateShift = 30;
constexpr StateWord kStateMask = 3u << kStateShift;
constexpr StateWord kReservedMask = 1u << 29;
constexpr StateWord kInRingMask = 1u << 28;
constexpr StateWord kUsedMask = 1u << 27;
constexpr StateWord kRefMask = (1u << 27) - 1u;
constexpr uint32_t kRefMax = kRefMask;

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

__host__ __device__ inline NodeState state_of(StateWord w) {
  return static_cast<NodeState>((w & kStateMask) >> kStateShift);
}
__host__ __device__ inline uint32_t ref_of(StateWord w) { return w & kRefMask; }
__host__ __device__ inline bool in_ring_of(StateWord w) {
  return (w & kInRingMask) != 0u;
}
__host__ __device__ inline bool used_of(StateWord w) {
  return (w & kUsedMask) != 0u;
}

__host__ __device__ inline StateWord make_state(NodeState s, uint32_t ref,
                                                bool in_ring, bool used) {
  return (static_cast<StateWord>(s) << kStateShift) |
         (in_ring ? kInRingMask : 0u) | (used ? kUsedMask : 0u) |
         (ref & kRefMask);
}

// Every node starts EMPTY with in_ring set: the ring is initialised to
// arange(C), so each node is already sitting in it.
__host__ __device__ inline StateWord initial_state() {
  return make_state(kEmpty, 0u, /*in_ring=*/true, /*used=*/false);
}

// ---------------------------------------------------------------------------
// Transitions
// ---------------------------------------------------------------------------

// Pin a node on a cache hit: ref+1 and set `used`.  Fails if the node is not
// RESIDENT, i.e. if an allocator claimed it first — the caller then treats the
// key as a miss rather than waiting.  Spinning here would mean spinning on an
// operation that may contain a PCIe transfer.
//
// `used` is folded into the same CAS, so CLOCK's recency bit costs nothing:
// no extra memory traffic on the hot path.
//
// NOTE: the caller must re-check the node's key after a successful pin.  The
// node may have been evicted and reassigned to a different key between reading
// the pointer to it and this CAS; the state would read RESIDENT again and the
// pin would wrongly succeed (INV-8).
__device__ inline bool try_pin(StateWord* state) {
  StateWord old = *reinterpret_cast<volatile StateWord*>(state);
  for (;;) {
    if (state_of(old) != kResident) return false;
    // Guard the carry out of the ref field into `used`.  27 bits is far more
    // than any real pipeline depth, so hitting this means ref is leaking.
    if (ref_of(old) == kRefMax) {
      LW_CACHE_ASSERT(false && "lw_cache: ref_counter overflow");
      return false;
    }
    const StateWord want = (old | kUsedMask) + 1u;
    const StateWord prev = atomicCAS(state, old, want);
    if (prev == old) return true;
    old = prev;
  }
}

// Claim a node for reuse.  Succeeds only for an idle node: RESIDENT or EMPTY
// with ref == 0.  This is INV-1 and INV-3 — a node with outstanding references
// can never be taken, and the check is redone under CAS so it cannot race with
// a concurrent pin.
//
// Other bits are preserved; the caller clears in_ring separately.
//
// On success `*claimed_from` receives the word as it was immediately before the
// claim.  The caller needs it to tell an EMPTY node (nothing to write back)
// from one holding a row — see needs_writeback().
__device__ inline bool try_claim(StateWord* state, StateWord* claimed_from) {
  StateWord old = *reinterpret_cast<volatile StateWord*>(state);
  for (;;) {
    const NodeState s = state_of(old);
    if ((s != kResident && s != kEmpty) || ref_of(old) != 0u) return false;
    const StateWord want =
        (old & ~kStateMask) | (static_cast<StateWord>(kEvicting) << kStateShift);
    const StateWord prev = atomicCAS(state, old, want);
    if (prev == old) {
      if (claimed_from != nullptr) *claimed_from = old;
      return true;
    }
    old = prev;
  }
}

// Whether a claimed node holds data that must be written back before reuse.
// EMPTY nodes never have; everything else has, unconditionally — there is no
// dirty bit, because a resident row is by construction no older than its
// backing copy (INV-9), so the write is always "newer over older".
__host__ __device__ inline bool needs_writeback(StateWord claimed_from) {
  return state_of(claimed_from) != kEmpty;
}

// Bind a claimed node to its new key and pin it for the requesting step.
// EVICTING -> LOADING, ref = 1, used cleared (a fresh arrival has not proved
// itself yet — see CLOCK), in_ring cleared (the ring entry was consumed).
__host__ __device__ inline StateWord loading_state() {
  return make_state(kLoading, 1u, /*in_ring=*/false, /*used=*/false);
}

// LOADING -> RESIDENT once the row's contents are valid.
__device__ inline void mark_resident(StateWord* state) {
  StateWord old = *reinterpret_cast<volatile StateWord*>(state);
  for (;;) {
    LW_CACHE_ASSERT(state_of(old) == kLoading);
    const StateWord want =
        (old & ~kStateMask) | (static_cast<StateWord>(kResident) << kStateShift);
    const StateWord prev = atomicCAS(state, old, want);
    if (prev == old) return;
    old = prev;
  }
}

// Release a reference at the end of the backward pass.
//
// Returns true when this thread is the one that must push the node onto the
// ring: it observed the 1 -> 0 transition *and* won the in_ring flag.  Both
// conditions matter.  The transition alone would let two steps releasing the
// same node race; the flag alone would re-enqueue a node that is already
// listed, and a node listed twice is a use-after-free waiting to happen — the
// first entry gets recycled, the second dangles onto whatever key now lives
// there.
__device__ inline bool release_ref(StateWord* state) {
  const StateWord old = atomicSub(state, 1u);
  // A word-wide decrement only means "ref -= 1" while ref > 0.  At zero it
  // borrows into used / in_ring / state.
  LW_CACHE_ASSERT(ref_of(old) > 0u &&
                  "lw_cache: unbalanced release; state word corrupted");
  if (ref_of(old) != 1u) return false;
  const StateWord prev = atomicOr(state, kInRingMask);
  return !in_ring_of(prev);
}

// Drop the ring-membership flag.  Called when a ring entry is consumed or
// discarded, so the node becomes eligible to be enqueued again on its next
// release.  Discarding without clearing this would strand the node outside the
// ring forever.
__device__ inline void clear_in_ring(StateWord* state) {
  atomicAnd(state, ~kInRingMask);
}

// CLOCK: test and clear the recency bit.  True means the node was hit since the
// hand last swept it and has earned a second chance.
__device__ inline bool take_used(StateWord* state) {
  const StateWord prev = atomicAnd(state, ~kUsedMask);
  return used_of(prev);
}

}  // namespace lw_cache
