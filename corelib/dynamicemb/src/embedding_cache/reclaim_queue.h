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
// The queue of nodes that may be taken for reuse.
//
// Initialised to arange(capacity), which collapses two structures into one:
// there is no free list of never-used slots alongside an eviction queue of used
// ones. Everything in here is simply "a node nobody is referencing". Some have
// never held data; some hold a valid row whose last user has finished, and
// those are still cache hits right up until the moment they are popped. The
// allocator does not care which it gets.
//
// Which is why this is not an eviction queue: half of what it holds has nothing
// to evict.
//
// ---------------------------------------------------------------------------
// The constraint that makes this simple
// ---------------------------------------------------------------------------
//
// At any moment this queue is touched by popping kernels or by pushing
// kernels, never both. Stream ordering enforces it.
//
// Without that it is the textbook MPMC ring hazard: a pusher bumps `tail` to
// reserve a slot, and before it stores the node id a popper reads `tail`, sees
// that slot as available, and consumes whatever was in it last time round.
// Fixing it properly wants a per-slot sequence number, which is a large step up
// in complexity and in how hard it is to tell when it is wrong.
//
// The hazard is one-directional, which is why one event dependency suffices:
//
//   pop running, push starts   — safe. The pusher writes at a position >= the
//                                popper's tail snapshot; the popper only reads
//                                below it.
//   push in flight, pop starts — unsafe. Exactly the window above.
//
// So an allocating kernel must wait for every already-issued releasing kernel.
// The reverse costs nothing and is not required. This holds at any prefetch
// depth; a deeper pipeline only means a released node becomes available later,
// which is a capacity question rather than a correctness one.
//
// Because pushes cannot begin mid-pop, `tail` does not move for the duration of
// a popping kernel: read it once into a snapshot and the bound check is exact
// and free for the rest of the kernel.
//
// ---------------------------------------------------------------------------
// FIFO only
// ---------------------------------------------------------------------------
//
// No second-chance policy here. CLOCK needs somewhere to put a node it has
// spared — it cannot go straight back on the queue, because writing `tail` from
// inside a popping kernel is exactly the overlap ruled out above — so it needs
// a landing buffer and a follow-up kernel to drain it. That machinery is where
// the nastiest defect in the previous implementation lived, and it misbehaved
// only at full speed, never under a race checker.
//
// Nothing is lost by waiting. NodeState keeps its `used` bit, set for free
// inside the pin, and the fraction of popped nodes carrying it is exactly where
// CLOCK's benefit would come from — measurable while running FIFO, and it
// already excludes the pinned nodes that stale-dropping protects under either
// policy. Build it when that number says to.
// ---------------------------------------------------------------------------

#pragma once

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "allocator.h"
#include "ec_assert.h"
#include "launch.h"
#include "reserve.h"

#ifndef __CUDACC__
#error \
    "embedding_cache/reclaim_queue.h launches kernels and must be compiled by \
nvcc. Including it from a .cpp will not work; make the translation unit a .cu."
#endif

namespace embedding_cache {

namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// ReclaimQueue — the view, passed to kernels by value
// ---------------------------------------------------------------------------

template <typename NodeIdT = int32_t,
          NodeIdT kNoNodeV = static_cast<NodeIdT>(-1)>
struct ReclaimQueue {
  using NodeId = NodeIdT;

  // Returned by pop() when the queue is exhausted, and it has to be the same
  // value HashMap uses for "no node": the two travel together through a caller
  // that cannot tell whose it is holding.
  //
  // Parameters rather than fixed constants so that agreement can be derived
  // instead of restated —
  //
  //   using Queue = ReclaimQueue<Map::NodeId, Map::kNoNode>;
  //
  // which is one place to change instead of two to keep in step.
  static constexpr NodeId kNoNode = kNoNodeV;

  static_assert(std::is_integral<NodeIdT>::value,
                "NodeId must be integral: it indexes the node pool");

  // -- state, all borrowed --------------------------------------------------

  NodeId* slots;  // [capacity]

  // Monotonic, and 64-bit so they never wrap in any real run. Counters rather
  // than wrapped indices, because then `tail - head` is the live count with no
  // ambiguity between full and empty.
  uint64_t* head;
  uint64_t* tail;

  // The node count, and the length of `slots`. Not required to be a power of
  // two — it is whatever the cache was configured with — so indexing is a
  // modulo rather than a mask. That costs a 64-bit division, which has no
  // native instruction on any of these architectures, on every push and on
  // every iteration of a pop.
  //
  // Worth it for what it avoids: rounding the slot array up to a power of two
  // would mean storing the rounded length as well, and two fields that have to
  // agree are two fields that can disagree. The division lands next to an
  // atomic that costs a trip to L2, and the whole operation runs once per
  // unique key per step. If a profile ever says otherwise, the answer is a
  // precomputed reciprocal, and that is the point at which a second field earns
  // its place.
  int64_t capacity;

  // -- block-collective scratch ---------------------------------------------
  //
  // One per block, in shared memory:
  //
  //   __shared__ ReclaimQueue<>::Scratch scratch;
  //
  // Aliased rather than redeclared, because reserving a slot on this queue and
  // reserving one in a compaction output are the same operation — see
  // reserve.h. Left uninitialised on purpose: reserve() zeroes it.
  using Scratch = BlockReserve::Scratch;

  // -- the interface --------------------------------------------------------
  //
  // BOTH ARE BLOCK-COLLECTIVE. Every thread in the block has to call them, the
  // same number of times, in the same order — they contain __syncthreads(), and
  // a thread that skips one hangs the block. `want` is how a thread that has
  // nothing to do still takes part:
  //
  //   for (int64_t base = blockIdx.x * blockDim.x; base < n;
  //        base += gridDim.x * blockDim.x) {
  //     const int64_t i = base + threadIdx.x;
  //     queue.push(i < n && owns_enqueue(i), node_of(i), scratch);
  //   }
  //
  // Note the shape: the loop bound is the same for every thread, and the tail
  // of the range is handled by `want` rather than by exiting early. This is the
  // same discipline unique_op.cu's segmented_unique_core uses, for the same
  // reason.

  // Enqueue a node whose last reference just went away.
  //
  // Pass want=true only when NodeState::unpin said this thread owns the enqueue
  // — that call, not this one, is what decides ownership, and it decides it in
  // the same atomic that took the count to zero.
  //
  // No overflow check, and none can be needed: the in_queue bit admits at most
  // one entry per node, so live entries never exceed the node count.
  //
  // Aggregated two levels deep — see reserve.h. Up to 256 threads end up behind
  // one atomic on `tail`, which matters because every one of them targets the
  // same address, and same-address atomics serialise in L2.
  __forceinline__ __device__ void push(bool want, NodeId node, Scratch& s) {
    const uint64_t pos = BlockReserve::reserve(
        want, reinterpret_cast<unsigned long long*>(tail), s);
    if (want) slots[pos % static_cast<uint64_t>(capacity)] = node;
  }

  // Read the bound for a popping kernel. Valid for the whole kernel; see the
  // header comment for why.
  [[nodiscard]] __forceinline__ __device__ uint64_t snapshot_tail() const {
    return *reinterpret_cast<const volatile uint64_t*>(tail);
  }

  // Take a node for reuse, or report the queue exhausted.
  //
  // `try_take(node)` decides whether a candidate can be had — in practice
  // AtomicNodeState::try_claim, which refuses a node referenced again since it
  // was enqueued. Passing it in rather than calling it directly is what keeps
  // this file from knowing what a node's state looks like: the retry loop is
  // queue logic, the decision is not.
  //
  // A refused entry is dropped, never requeued. Requeueing would livelock the
  // moment every node is pinned — threads would circle forever finding nothing
  // takeable. Dropping shrinks the queue monotonically until either a node is
  // taken or it runs dry and the caller gets a definite failure. Nothing leaks:
  // try_claim leaves in_queue clear, so the node re-enqueues on its next
  // release.
  //
  // Exhaustion means every node is pinned by an in-flight step, which is a
  // capacity planning error. It is reported, never worked around quietly.
  // Only the first ticket is aggregated. A thread whose candidate is refused
  // needs another one, and how many it needs is not known to anybody else, so
  // the retries cannot be block-collective — a barrier inside that loop would
  // deadlock the moment two threads wanted different numbers of rounds. They
  // fall back to one global atomic each, which is what the old shape did for
  // every ticket.
  //
  // That is the right split: the first attempt succeeds unless the entry is
  // stale, so the aggregated path carries the common case and the unaggregated
  // one carries the exception.
  template <typename TryTake>
  [[nodiscard]] __forceinline__ __device__ NodeId pop(bool want,
                                                      uint64_t tail_snapshot,
                                                      Scratch& s,
                                                      TryTake&& try_take) {
    const uint64_t first = BlockReserve::reserve(
        want, reinterpret_cast<unsigned long long*>(head), s);

    NodeId node = kNoNode;
    bool settled = !want;

    if (want) {
      const uint64_t pos = first;
      if (pos >= tail_snapshot) {
        // Past the bound. Hand the position back, aggregated the same way, or
        // `head` runs past `tail` and the live count underflows. CUDA has no
        // 64-bit atomicSub — only int and unsigned int — so this adds the two's
        // complement instead.
        //
        // `head` cannot end below the snapshot: only a thread that pushed it
        // above brings it back down, and each hands back exactly what it took.
        auto g = cg::coalesced_threads();
        if (g.thread_rank() == 0) {
          atomicAdd(reinterpret_cast<unsigned long long*>(head),
                    ~static_cast<unsigned long long>(g.size()) + 1ull);
        }
        settled = true;  // exhausted, and kNoNode is the answer
      } else {
        const NodeId candidate = slots[pos % static_cast<uint64_t>(capacity)];
        if (try_take(candidate)) {
          node = candidate;
          settled = true;
        }
      }
    }

    // Retries: one thread, one atomic, no barrier.
    while (!settled) {
      const uint64_t pos =
          atomicAdd(reinterpret_cast<unsigned long long*>(head), 1ULL);
      if (pos >= tail_snapshot) {
        atomicAdd(reinterpret_cast<unsigned long long*>(head), ~0ULL);
        break;
      }
      const NodeId candidate = slots[pos % static_cast<uint64_t>(capacity)];
      if (try_take(candidate)) {
        node = candidate;
        break;
      }
    }
    return node;
  }

  // How many entries are live. A hint: with pushes in flight on another stream
  // it is stale by the time it is read. For metrics, never for a decision.
  [[nodiscard]] __forceinline__ __host__ __device__ int64_t size_hint(
      uint64_t h, uint64_t t) const {
    return static_cast<int64_t>(t - h);
  }
};

// ---------------------------------------------------------------------------
// Initialisation kernel
// ---------------------------------------------------------------------------

namespace detail {

// A template rather than a plain __global__ so that it can live in a header
// without nvcc emitting a duplicate symbol in every translation unit that
// includes it. (`inline` on a non-template __global__ is ignored, with a
// warning that is easy to miss.)
template <typename NodeIdT>
__global__ void queue_arange_kernel(NodeIdT* slots, int64_t capacity) {
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < capacity;
       i += int64_t(gridDim.x) * blockDim.x) {
    slots[i] = static_cast<NodeIdT>(i);
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// ReclaimQueueStorage
// ---------------------------------------------------------------------------
//
// Owns the slot array and the two counters.
//
// Unlike the node pool there is nothing here belonging to another component, so
// this one can initialise everything it allocates: arange for the slots, zero
// for head, capacity for tail. A freshly built queue is a complete queue.

template <typename NodeIdT = int32_t,
          NodeIdT kNoNodeV = static_cast<NodeIdT>(-1),
          typename AllocT = CudaAllocator>
class ReclaimQueueStorage {
 public:
  using Queue = ReclaimQueue<NodeIdT, kNoNodeV>;
  using NodeId = NodeIdT;
  using Allocator = AllocT;
  using Buffer = typename AllocT::Buffer;

  static constexpr NodeId kNoNode = Queue::kNoNode;

  static_assert(std::is_trivially_copyable<Queue>::value,
                "the queue view reaches kernels by value");

  // `stream` is kept rather than passed per call, and the constructor is
  // asynchronous with respect to it: it enqueues the arange and returns. The
  // download_* diagnostics are ordered on the same stream and synchronise, so
  // they stay safe to call immediately.
  explicit ReclaimQueueStorage(int64_t capacity, AllocT allocator = AllocT{},
                               cudaStream_t stream = nullptr)
      : capacity_(capacity), stream_(stream), alloc_(std::move(allocator)) {
    if (capacity <= 0) {
      throw std::invalid_argument("ReclaimQueueStorage: capacity must be > 0");
    }
    if (capacity > static_cast<int64_t>(INT32_MAX)) {
      throw std::invalid_argument(
          "ReclaimQueueStorage: capacity exceeds the 32-bit node id range");
    }
    slots_ = alloc_.allocate(sizeof(NodeId) * size_t(capacity_));
    // Head and tail in one allocation: they are read together by every kernel
    // that touches the queue, and two four-byte-apart counters share a line.
    counters_ = alloc_.allocate(sizeof(uint64_t) * 2u);
    reset();
  }

  ReclaimQueueStorage(const ReclaimQueueStorage&) = delete;
  ReclaimQueueStorage& operator=(const ReclaimQueueStorage&) = delete;

  [[nodiscard]] Queue queue() const {
    return Queue{d_slots(), d_head(), d_tail(), capacity_};
  }

  // Every node queued, in order, nothing taken yet.
  //
  // The caller has to put the node pool into the state that matches — every
  // node with in_queue set — or the two disagree and a node ends up queued
  // twice. That pairing cannot happen here: this component does not know what a
  // node looks like.
  void reset() {
    detail::queue_arange_kernel<<<grid_for(capacity_), kBlock, 0, stream_>>>(
        d_slots(), capacity_);
    check(cudaGetLastError(), "arange launch");

    const uint64_t counters[2] = {0u, static_cast<uint64_t>(capacity_)};
    check(cudaMemcpyAsync(d_head(), counters, sizeof(counters),
                          cudaMemcpyHostToDevice, stream_),
          "counter upload");
    check(cudaStreamSynchronize(stream_), "counter upload sync");
  }

  [[nodiscard]] int64_t capacity() const { return capacity_; }
  [[nodiscard]] cudaStream_t stream() const { return stream_; }
  [[nodiscard]] const Allocator& allocator() const { return alloc_; }

  [[nodiscard]] size_t bytes() const {
    return sizeof(NodeId) * size_t(capacity_) + sizeof(uint64_t) * 2u;
  }

  // -- inspection -----------------------------------------------------------

  [[nodiscard]] std::vector<NodeId> download_slots() const {
    return download(d_slots(), size_t(capacity_));
  }

  // {head, tail}.
  [[nodiscard]] std::vector<uint64_t> download_counters() const {
    return download(d_head(), 2u);
  }

  // Entries currently live. Only meaningful when nothing is running.
  [[nodiscard]] int64_t size() const {
    const auto c = download_counters();
    return static_cast<int64_t>(c[1] - c[0]);
  }

 private:
  static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("ReclaimQueueStorage: ") + what +
                               ": " + cudaGetErrorString(e));
    }
  }

  template <typename T>
  std::vector<T> download(const T* src, size_t n) const {
    std::vector<T> out(n);
    check(cudaMemcpyAsync(out.data(), src, n * sizeof(T),
                          cudaMemcpyDeviceToHost, stream_),
          "download");
    check(cudaStreamSynchronize(stream_), "download sync");
    return out;
  }

  NodeId* d_slots() const { return static_cast<NodeId*>(slots_.data()); }
  uint64_t* d_head() const { return static_cast<uint64_t*>(counters_.data()); }
  uint64_t* d_tail() const { return d_head() + 1; }

  int64_t capacity_ = 0;
  cudaStream_t stream_ = nullptr;
  AllocT alloc_;
  Buffer slots_, counters_;
};

}  // namespace embedding_cache
