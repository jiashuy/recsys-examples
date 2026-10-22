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
// Launches, and the stream discipline the design depends on.
//
// Kept apart from kernels.cuh because <<<>>> needs a CUDA compiler, while the
// kernel bodies do not — that separation is what lets the bodies be exercised
// by testing/run_host_tests.sh. Nothing in this file is covered by those tests.
//
// ---------------------------------------------------------------------------
// The one constraint that cannot be expressed by ordering alone
// ---------------------------------------------------------------------------
//
// Within a step the phases are ordered by being separate launches on one
// stream. Across steps, with prefetch running ahead, one pair is not:
//
//   allocate pops the ring; release pushes it. A pop kernel must not start
//   while a push is in flight, because a pusher reserves a slot with
//   atomicAdd(tail) before storing the node id into it, and a pop that
//   snapshots `tail` in that window will treat the reserved slot as readable
//   and consume whatever it held last.
//
// The hazard is one-directional. A push that starts *during* a pop is
// harmless: it writes at or beyond the popper's snapshot, which the popper
// never reads. So a single event, waited on in one direction, is enough —
// `Launcher::allocate` waits for the most recent release; release never waits.
//
// In steady state this costs nothing. At prefetch depth d, allocate for step
// N+d runs around the forward pass of step N, by which time release for step
// N-1 finished long ago. Deeper pipelines only mean a released node takes
// longer to become available again, which is a capacity question (see
// CacheConfig validation), not a correctness one.
//
// All prefetch work must share one stream. Two prefetch streams would let
// lookup for one step overlap allocate for another, and a node found by lookup
// could then be claimed and rebound before the pin lands — the pin would see
// RESIDENT again and succeed on the wrong key.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "kernels.cuh"
#include "multi_kernels.cuh"

namespace lw_cache {

inline void cuda_check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string("lw_cache: ") + what + ": " +
                             cudaGetErrorString(e));
  }
}

constexpr int kDefaultBlock = 256;
constexpr int kLanesPerRow = 32;  // one warp per row

inline int grid_for(int64_t items, int block = kDefaultBlock) {
  if (items <= 0) return 1;
  const int64_t g = (items + block - 1) / block;
  // Cap the grid: the kernels are grid-stride loops, so a smaller grid is only
  // a scheduling choice, never a correctness one.
  const int64_t kMaxBlocks = 65535;
  return static_cast<int>(g < kMaxBlocks ? g : kMaxBlocks);
}

inline int grid_for_rows(int64_t rows, int block = kDefaultBlock) {
  return grid_for(rows * kLanesPerRow, block);
}

class Launcher {
 public:
  Launcher() {
    cuda_check(cudaEventCreateWithFlags(&release_done_, cudaEventDisableTiming),
               "cudaEventCreate");
    // Recorded once so the first allocate has something to wait on.
    cuda_check(cudaEventRecord(release_done_, 0), "cudaEventRecord (init)");
  }
  ~Launcher() { cudaEventDestroy(release_done_); }

  Launcher(const Launcher&) = delete;
  Launcher& operator=(const Launcher&) = delete;

  template <typename ValueT>
  void lookup_pin(const CacheView<ValueT>& view, const Key* keys,
                  int64_t num_keys, int32_t* out_nodes, cudaStream_t stream) {
    if (num_keys <= 0) return;
    lookup_pin_kernel<<<grid_for(num_keys), kDefaultBlock, 0, stream>>>(
        view, keys, num_keys, out_nodes);
    cuda_check(cudaGetLastError(), "lookup_pin_kernel");
  }

  // Issues the whole allocate sequence. `tail_snapshot` must have been read
  // after the wait below, so it cannot include a slot still being written.
  template <typename ValueT, typename BackingT, typename Init>
  void allocate(const CacheView<ValueT>& view, const BackingT& backing,
                AllocSlot* slots, int64_t num_slots, uint64_t tail_snapshot,
                const PopPolicy& policy, const RetryBuffer& retry,
                bool use_retry, const Init& init, cudaStream_t stream) {
    if (num_slots <= 0) return;

    // The one cross-step dependency; see the header comment.
    cuda_check(cudaStreamWaitEvent(stream, release_done_, 0),
               "cudaStreamWaitEvent(release)");

    const int g = grid_for(num_slots);
    const int gr = grid_for_rows(num_slots);

    claim_kernel<<<g, kDefaultBlock, 0, stream>>>(view, backing, slots,
                                                  num_slots, tail_snapshot,
                                                  policy, retry, use_retry);
    cuda_check(cudaGetLastError(), "claim_kernel");

    if (use_retry) {
      absorb_retry_kernel<<<g, kDefaultBlock, 0, stream>>>(view, retry);
      cuda_check(cudaGetLastError(), "absorb_retry_kernel");
      // Must happen, and must happen here rather than inside the kernel. A
      // count that survives into the next step makes it re-push nodes that are
      // already queued, which is a duplicate ring entry and from there a node
      // handed to two threads at once.
      cuda_check(cudaMemsetAsync(retry.count, 0, sizeof(uint32_t), stream),
                 "reset retry count");
    }

    // Two passes, not one: removal and insertion in the same kernel interleave
    // on a shared bucket.
    unlink_kernel<<<g, kDefaultBlock, 0, stream>>>(view, backing, slots,
                                                   num_slots);
    cuda_check(cudaGetLastError(), "unlink_kernel");

    bind_kernel<<<g, kDefaultBlock, 0, stream>>>(view, slots, num_slots);
    cuda_check(cudaGetLastError(), "bind_kernel");

    // P0: the two directions are separate launches. P1 merges them so both
    // travel at once, once the link is measured to be full duplex.
    writeback_kernel<kLanesPerRow>
        <<<gr, kDefaultBlock, 0, stream>>>(view, backing, slots, num_slots);
    cuda_check(cudaGetLastError(), "writeback_kernel");

    fill_kernel<kLanesPerRow>
        <<<gr, kDefaultBlock, 0, stream>>>(view, backing, slots, num_slots, init);
    cuda_check(cudaGetLastError(), "fill_kernel");
  }

  // Same as allocate(), but the slot count lives on the device: the caller
  // compacted its misses with a kernel and never learned how many there were.
  //
  // That matters beyond saving a copy. With the count on the host there has to
  // be a synchronisation between lookup and allocate, and that sync silently
  // orders allocate against release — which is what the stream event below is
  // for. Keep the sync and the event is never exercised; drop it and the event
  // has to be right.
  //
  // The kernels are launched over the worst case (`max_slots`) and each thread
  // reads the real count, so nothing on the host needs to know it.
  template <typename ValueT, typename BackingT, typename Init>
  void allocate_compacted(const CacheView<ValueT>& view, const BackingT& backing,
                          AllocSlot* slots, const uint32_t* d_count,
                          int64_t max_slots, uint64_t tail_snapshot,
                          const PopPolicy& policy, const RetryBuffer& retry,
                          bool use_retry, const Init& init,
                          cudaStream_t stream) {
    if (max_slots <= 0) return;

    cuda_check(cudaStreamWaitEvent(stream, release_done_, 0),
               "cudaStreamWaitEvent(release)");

    const int g = grid_for(max_slots);
    const int gr = grid_for_rows(max_slots);

    claim_kernel_n<<<g, kDefaultBlock, 0, stream>>>(
        view, backing, slots, d_count, tail_snapshot, policy, retry, use_retry);
    cuda_check(cudaGetLastError(), "claim_kernel_n");

    if (use_retry) {
      absorb_retry_kernel<<<g, kDefaultBlock, 0, stream>>>(view, retry);
      cuda_check(cudaGetLastError(), "absorb_retry_kernel");
      cuda_check(cudaMemsetAsync(retry.count, 0, sizeof(uint32_t), stream),
                 "reset retry count");
    }

    unlink_kernel_n<<<g, kDefaultBlock, 0, stream>>>(view, backing, slots,
                                                     d_count);
    cuda_check(cudaGetLastError(), "unlink_kernel_n");

    bind_kernel_n<<<g, kDefaultBlock, 0, stream>>>(view, slots, d_count);
    cuda_check(cudaGetLastError(), "bind_kernel_n");

    writeback_kernel_n<kLanesPerRow>
        <<<gr, kDefaultBlock, 0, stream>>>(view, backing, slots, d_count);
    cuda_check(cudaGetLastError(), "writeback_kernel_n");

    fill_kernel_n<kLanesPerRow>
        <<<gr, kDefaultBlock, 0, stream>>>(view, backing, slots, d_count, init);
    cuda_check(cudaGetLastError(), "fill_kernel_n");
  }

  template <typename ValueT>
  void release(const CacheView<ValueT>& view, const int32_t* nodes,
               int64_t num_nodes, cudaStream_t stream) {
    if (num_nodes > 0) {
      release_kernel<<<grid_for(num_nodes), kDefaultBlock, 0, stream>>>(
          view, nodes, num_nodes);
      cuda_check(cudaGetLastError(), "release_kernel");
    }
    // Every subsequent allocate waits on this.
    cuda_check(cudaEventRecord(release_done_, stream), "cudaEventRecord");
  }

  // Checkpoint. Drains first: reading a row concurrently with the optimizer
  // gives a torn row, and concurrently with an allocate gives another key's
  // data. Draining removes both and makes the reference count check exact.
  template <typename ValueT, typename BackingT>
  void flush(const CacheView<ValueT>& view, const BackingT& backing,
             cudaStream_t stream) {
    cuda_check(cudaStreamSynchronize(stream), "drain before flush");
    flush_kernel<kLanesPerRow>
        <<<grid_for_rows(view.capacity), kDefaultBlock, 0, stream>>>(view,
                                                                    backing);
    cuda_check(cudaGetLastError(), "flush_kernel");
  }

  // Number of nodes still referenced. Zero after a drain; anything else is an
  // INV-6 leak, which otherwise only surfaces as a hit rate that sags over
  // hours with no obvious cause. A checkpoint is a natural place to check.
  template <typename ValueT>
  unsigned long long count_referenced(const CacheView<ValueT>& view,
                                      unsigned long long* d_scratch,
                                      cudaStream_t stream) {
    cuda_check(cudaMemsetAsync(d_scratch, 0, sizeof(*d_scratch), stream),
               "memset scratch");
    count_referenced_kernel<<<grid_for(view.capacity), kDefaultBlock, 0,
                              stream>>>(view, d_scratch);
    cuda_check(cudaGetLastError(), "count_referenced_kernel");
    unsigned long long out = 0;
    cuda_check(cudaMemcpyAsync(&out, d_scratch, sizeof(out),
                               cudaMemcpyDeviceToHost, stream),
               "copy scratch");
    cuda_check(cudaStreamSynchronize(stream), "sync scratch");
    return out;
  }

  // -- multi-table -------------------------------------------------------
  //
  // Identical phase order and identical event discipline; the kernels differ
  // only in how a thread finds its table.

  template <typename ValueT>
  void mt_lookup_pin(const MultiCacheView<ValueT>& mv, const Key* keys,
                     const int64_t* table_range, int64_t num_keys,
                     int32_t* out_nodes, cudaStream_t stream) {
    if (num_keys <= 0) return;
    mt_lookup_pin_kernel<<<grid_for(num_keys), kDefaultBlock, 0, stream>>>(
        mv, keys, table_range, num_keys, out_nodes);
    cuda_check(cudaGetLastError(), "mt_lookup_pin_kernel");
  }

  // `tail_snapshots` is a device array with one entry per table, read after the
  // wait below for the same reason the single-table path reads one.
  template <typename ValueT, typename BackingT, typename Init>
  void mt_allocate(const MultiCacheView<ValueT>& mv,
                   const MultiBackingView<BackingT>& backing, AllocSlot* slots,
                   int64_t num_slots, const uint64_t* tail_snapshots,
                   const PopPolicy& policy, bool use_retry, const Init& init,
                   cudaStream_t stream) {
    if (num_slots <= 0) return;

    cuda_check(cudaStreamWaitEvent(stream, release_done_, 0),
               "cudaStreamWaitEvent(release)");

    const int g = grid_for(num_slots);
    const int gr = grid_for_rows(num_slots);

    mt_claim_kernel<<<g, kDefaultBlock, 0, stream>>>(
        mv, backing, slots, num_slots, tail_snapshots, policy, use_retry);
    cuda_check(cudaGetLastError(), "mt_claim_kernel");

    if (use_retry) {
      // One thread per table; each drains its own buffer.
      mt_absorb_retry_kernel<<<1, kDefaultBlock, 0, stream>>>(mv);
      cuda_check(cudaGetLastError(), "mt_absorb_retry_kernel");
    }

    mt_unlink_kernel<<<g, kDefaultBlock, 0, stream>>>(mv, backing, slots,
                                                      num_slots);
    cuda_check(cudaGetLastError(), "mt_unlink_kernel");

    mt_bind_kernel<<<g, kDefaultBlock, 0, stream>>>(mv, slots, num_slots);
    cuda_check(cudaGetLastError(), "mt_bind_kernel");

    mt_writeback_kernel<kLanesPerRow>
        <<<gr, kDefaultBlock, 0, stream>>>(mv, backing, slots, num_slots);
    cuda_check(cudaGetLastError(), "mt_writeback_kernel");

    mt_fill_kernel<kLanesPerRow>
        <<<gr, kDefaultBlock, 0, stream>>>(mv, backing, slots, num_slots, init);
    cuda_check(cudaGetLastError(), "mt_fill_kernel");
  }

  // Multi-table counterpart of allocate_compacted(): slot count on the device,
  // so no synchronisation sits between lookup and allocate.
  template <typename ValueT, typename BackingT, typename Init>
  void mt_allocate_compacted(const MultiCacheView<ValueT>& mv,
                             const MultiBackingView<BackingT>& backing,
                             AllocSlot* slots, const uint32_t* d_count,
                             int64_t max_slots, const uint64_t* tail_snapshots,
                             const PopPolicy& policy, bool use_retry,
                             const Init& init, cudaStream_t stream) {
    if (max_slots <= 0) return;

    cuda_check(cudaStreamWaitEvent(stream, release_done_, 0),
               "cudaStreamWaitEvent(release)");

    const int g = grid_for(max_slots);
    const int gr = grid_for_rows(max_slots);

    mt_claim_kernel_n<<<g, kDefaultBlock, 0, stream>>>(
        mv, backing, slots, d_count, tail_snapshots, policy, use_retry);
    cuda_check(cudaGetLastError(), "mt_claim_kernel_n");

    if (use_retry) {
      mt_absorb_retry_kernel<<<1, kDefaultBlock, 0, stream>>>(mv);
      cuda_check(cudaGetLastError(), "mt_absorb_retry_kernel");
    }

    mt_unlink_kernel_n<<<g, kDefaultBlock, 0, stream>>>(mv, backing, slots,
                                                        d_count);
    cuda_check(cudaGetLastError(), "mt_unlink_kernel_n");

    mt_bind_kernel_n<<<g, kDefaultBlock, 0, stream>>>(mv, slots, d_count);
    cuda_check(cudaGetLastError(), "mt_bind_kernel_n");

    mt_writeback_kernel_n<kLanesPerRow>
        <<<gr, kDefaultBlock, 0, stream>>>(mv, backing, slots, d_count);
    cuda_check(cudaGetLastError(), "mt_writeback_kernel_n");

    mt_fill_kernel_n<kLanesPerRow>
        <<<gr, kDefaultBlock, 0, stream>>>(mv, backing, slots, d_count, init);
    cuda_check(cudaGetLastError(), "mt_fill_kernel_n");
  }

  template <typename ValueT>
  void mt_release(const MultiCacheView<ValueT>& mv, const int32_t* nodes,
                  const int64_t* table_range, int64_t num_nodes,
                  cudaStream_t stream) {
    if (num_nodes > 0) {
      mt_release_kernel<<<grid_for(num_nodes), kDefaultBlock, 0, stream>>>(
          mv, nodes, table_range, num_nodes);
      cuda_check(cudaGetLastError(), "mt_release_kernel");
    }
    cuda_check(cudaEventRecord(release_done_, stream), "cudaEventRecord");
  }

  // `node_range` maps a flat node index to its table, the same shape as
  // table_range but over nodes instead of keys.
  template <typename ValueT, typename BackingT>
  void mt_flush(const MultiCacheView<ValueT>& mv,
                const MultiBackingView<BackingT>& backing,
                const int64_t* node_range, int64_t total_nodes,
                cudaStream_t stream) {
    cuda_check(cudaStreamSynchronize(stream), "drain before flush");
    mt_flush_kernel<kLanesPerRow>
        <<<grid_for_rows(total_nodes), kDefaultBlock, 0, stream>>>(
            mv, backing, node_range, total_nodes);
    cuda_check(cudaGetLastError(), "mt_flush_kernel");
  }

  template <typename ValueT>
  unsigned long long mt_count_referenced(const MultiCacheView<ValueT>& mv,
                                         const int64_t* node_range,
                                         int64_t total_nodes,
                                         unsigned long long* d_scratch,
                                         cudaStream_t stream) {
    cuda_check(cudaMemsetAsync(d_scratch, 0, sizeof(*d_scratch), stream),
               "memset scratch");
    mt_count_referenced_kernel<<<grid_for(total_nodes), kDefaultBlock, 0,
                                 stream>>>(mv, node_range, total_nodes,
                                           d_scratch);
    cuda_check(cudaGetLastError(), "mt_count_referenced_kernel");
    unsigned long long out = 0;
    cuda_check(cudaMemcpyAsync(&out, d_scratch, sizeof(out),
                               cudaMemcpyDeviceToHost, stream),
               "copy scratch");
    cuda_check(cudaStreamSynchronize(stream), "sync scratch");
    return out;
  }

 private:
  cudaEvent_t release_done_{};
};

}  // namespace lw_cache
