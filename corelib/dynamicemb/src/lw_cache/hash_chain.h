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
// key -> node index, as buckets of doubly-linked nodes.
//
// The chain pointers live in the node arrays themselves (prev/next), so a
// bucket is a list of node ids and there is no separate entry object. Doubly
// linked rather than singly: eviction removes a node identified by id, not by
// position, and a singly-linked chain would have to be walked from its head to
// find the predecessor.
//
// ---------------------------------------------------------------------------
// Phases, and why insert and unlink must not share a kernel
// ---------------------------------------------------------------------------
//
// Three operations, each confined to its own kernel:
//
//   lookup  — the pin phase
//   unlink  — allocate, pass 4a (victims leave their buckets)
//   insert  — allocate, pass 4b (new keys join theirs)
//
// Splitting 4a from 4b is not tidiness. Run together, one thread removing a
// victim from bucket b and another pushing a new node onto the same bucket
// interleave on `heads[b]` and on the neighbours' prev/next — the general
// lock-free doubly-linked list problem, which wants marked pointers or
// per-bucket locks held across both operations. The victim set and the
// new-key set are disjoint by construction, so two passes dissolve the problem
// instead of solving it.
//
// ---------------------------------------------------------------------------
// Within a pass
// ---------------------------------------------------------------------------
//
// Two victims in one bucket still collide with each other, as do two inserts.
// Both take a short lock on the bucket head: swap in a sentinel, edit, store
// the new head back. Zero extra memory — the head doubles as the lock word —
// and with ~2C buckets against a few thousand victims per step, two threads
// wanting the same bucket is rare enough that the lock is uncontended in
// practice. It has to be correct anyway.
//
// Insert could be lock-free (a CAS loop pushing at the head), but keeping the
// back-pointer consistent then needs a fence and careful reasoning about a
// second thread's write to prev[]. One mechanism for both operations is worth
// more than saving an uncontended atomic.
//
// REQUIRES sm_70+: the lock spins, so it depends on independent thread
// scheduling for a warp where one lane holds the lock and others wait. On
// older architectures the waiting lanes would prevent the holder from
// progressing. The project targets 7.5 and up.
//
// VERIFICATION STATUS
//   verified   — lookup / insert / unlink over every position in a chain,
//                collision handling, head and tail removal, via
//                testing/run_host_tests.sh
//   unverified — the bucket lock under contention, and the claim that two
//                passes remove the insert/unlink interleaving. Both need a GPU.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "backing.h"
#include "hash.h"
#include "node_state.h"
#include "ring.h"  // kNoNode

namespace lw_cache {

// Held-lock marker for a bucket head. Distinct from kNoNode (-1), which is a
// legitimate value meaning "empty bucket".
constexpr int32_t kBucketLocked = -2;

struct HashChainView {
  int32_t* heads;     // [num_buckets]
  int32_t* prev;      // [capacity]
  int32_t* next;      // [capacity]
  Key* keys;          // [capacity]
  int64_t num_buckets;  // power of two
  int64_t bucket_mask;  // num_buckets - 1

  __host__ __device__ int64_t bucket_of(Key key) const {
    return static_cast<int64_t>(mix64(static_cast<uint64_t>(key)) &
                                static_cast<uint64_t>(bucket_mask));
  }
};

// ---------------------------------------------------------------------------
// Bucket lock
// ---------------------------------------------------------------------------

// Returns the head as it was before locking.
__device__ inline int32_t lock_bucket(const HashChainView& view, int64_t b) {
  for (;;) {
    const int32_t head = atomicExch(&view.heads[b], kBucketLocked);
    if (head != kBucketLocked) return head;
    // Another thread holds it; back off and retry.
    __threadfence();
  }
}

__device__ inline void unlock_bucket(const HashChainView& view, int64_t b,
                                     int32_t new_head) {
  __threadfence();
  atomicExch(&view.heads[b], new_head);
}

// ---------------------------------------------------------------------------
// Lookup — pin phase only, no locking
// ---------------------------------------------------------------------------

// Walk the bucket for `key`. Returns the node id, or kNoNode.
//
// Takes no lock: lookups run in a kernel where nothing mutates the chains. A
// locked head would therefore never be observed here, but the check costs one
// comparison and turns a phase-ordering mistake into a miss rather than a walk
// off a sentinel.
//
// The caller must pin the returned node and then re-check its key. Between
// this walk and that pin the node can be evicted and rebound to another key,
// and its state would read RESIDENT again (INV-8).
__device__ inline int32_t chain_lookup(const HashChainView& view, Key key) {
  const int64_t b = view.bucket_of(key);
  int32_t n = *reinterpret_cast<const volatile int32_t*>(&view.heads[b]);
  if (n == kBucketLocked) return kNoNode;
  while (n != kNoNode) {
    if (view.keys[n] == key) return n;
    n = *reinterpret_cast<const volatile int32_t*>(&view.next[n]);
  }
  return kNoNode;
}

// ---------------------------------------------------------------------------
// Insert — allocate pass 4b
// ---------------------------------------------------------------------------

// Push `node` onto the front of its bucket. The caller must have already
// written keys[node].
__device__ inline void chain_insert(const HashChainView& view, int32_t node) {
  const int64_t b = view.bucket_of(view.keys[node]);
  const int32_t head = lock_bucket(view, b);

  view.prev[node] = kNoNode;
  view.next[node] = head;
  if (head != kNoNode) view.prev[head] = node;

  unlock_bucket(view, b, node);
}

// ---------------------------------------------------------------------------
// Unlink — allocate pass 4a
// ---------------------------------------------------------------------------

// Remove `node` from its bucket. `key` is passed explicitly because the caller
// may already be overwriting keys[node] with the incoming key; the bucket is
// determined by the key the node is leaving, not the one it is joining.
__device__ inline void chain_unlink(const HashChainView& view, int32_t node, Key key) {
  const int64_t b = view.bucket_of(key);
  const int32_t head = lock_bucket(view, b);

  const int32_t p = view.prev[node];
  const int32_t n = view.next[node];

  if (p != kNoNode) {
    view.next[p] = n;
  }
  if (n != kNoNode) {
    view.prev[n] = p;
  }
  view.prev[node] = kNoNode;
  view.next[node] = kNoNode;

  // Removing the head moves it to the successor; otherwise it is unchanged.
  unlock_bucket(view, b, (head == node) ? n : head);
}

}  // namespace lw_cache
