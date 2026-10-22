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
// Counters.
//
// All device-resident, incremented with atomics, copied out asynchronously.
// Nothing here may put a synchronisation on the step path — a counter that
// stalls the pipeline to be read is worse than no counter.
//
// Disabled by leaving the pointer null, which turns every bump into a branch
// that is uniform across the warp and never taken.
//
// ---------------------------------------------------------------------------
// The one worth having
// ---------------------------------------------------------------------------
//
// `used_at_pop / evict_candidates` answers "if CLOCK were switched on, what
// fraction of evictions would it defer?" — and it answers it while running
// FIFO, because the recency bit is set on every hit regardless of policy and
// costs nothing to set (it rides along in the pin CAS).
//
// So the decision on whether CLOCK is worth its retry buffer and extra kernel
// comes out of an ordinary training run. No trace dump, no offline simulation,
// no implementing it first to find out.
//
// It is a screening number rather than a prediction: deferring one eviction
// changes which node is taken instead, and that second-order effect is not
// recoverable from this. Near zero means the working set has no reuse and CLOCK
// cannot help; forty per cent means FIFO is discarding hot rows at that rate.
//
// Note what it excludes, which is not obvious. A key being used *right now* has
// ref > 0, so its ring entry is dropped as stale before the recency bit is ever
// looked at — it is protected whatever the policy. So this counts only keys hot
// in recent steps but idle in this one. That is the correct denominator for the
// question "would CLOCK help", because it is exactly the set FIFO discards and
// CLOCK would not, but it does mean a workload whose hot keys appear in every
// single step will report zero opportunity while still having excellent reuse.
// Zero here means "CLOCK adds nothing", not "there is no locality".
//
// ---------------------------------------------------------------------------
// The one that is an alarm
// ---------------------------------------------------------------------------
//
// `ring_exhausted` should be exactly zero. Anything else means a step found
// every node pinned, which is a capacity planning error, not a slow path — the
// keys involved were not cached at all.
//
// VERIFICATION STATUS
//   verified   — counter arithmetic and every bump site, via
//                test/lw_cache/run_host_tests.sh
//   unverified — atomic contention on the counters, and the async copy-out.
//                Needs a GPU.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace lw_cache {

enum MetricId : int {
  // pin phase
  kMetricLookups = 0,       // keys looked up
  kMetricHits,              // resolved to a pinned node
  kMetricPinFailed,         // found in a chain but mid-eviction
  kMetricIdentityMismatch,  // pinned, then the key had changed (INV-8)

  // ring
  kMetricPopAttempts,     // entries examined
  kMetricStaleDropped,    // entry referred to a node that was hit again
  kMetricClaimLost,       // pinned between the check and the CAS
  kMetricEvictCandidates, // entries with ref == 0: the denominator below
  kMetricUsedAtPop,       // of those, how many carried the recency bit
  kMetricSecondChances,   // CLOCK deferrals actually granted
  kMetricRingExhausted,   // capacity error; must stay zero

  // allocate
  kMetricBackingFull,  // key could not be given a backing row
  kMetricFresh,        // no backing record: initialised, no transfer
  kMetricLoaded,       // pulled across from the backing
  kMetricWrittenBack,  // victim rows sent out

  kNumMetrics
};

inline const char* metric_name(int id) {
  switch (id) {
    case kMetricLookups: return "lookups";
    case kMetricHits: return "hits";
    case kMetricPinFailed: return "pin_failed";
    case kMetricIdentityMismatch: return "identity_mismatch";
    case kMetricPopAttempts: return "pop_attempts";
    case kMetricStaleDropped: return "stale_dropped";
    case kMetricClaimLost: return "claim_lost";
    case kMetricEvictCandidates: return "evict_candidates";
    case kMetricUsedAtPop: return "used_at_pop";
    case kMetricSecondChances: return "second_chances";
    case kMetricRingExhausted: return "ring_exhausted";
    case kMetricBackingFull: return "backing_full";
    case kMetricFresh: return "fresh";
    case kMetricLoaded: return "loaded";
    case kMetricWrittenBack: return "written_back";
    default: return "?";
  }
}

// ---------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------

struct MetricsView {
  unsigned long long* counters;  // [kNumMetrics], or null when disabled

  __device__ void bump(int id, unsigned long long n = 1ull) const {
    if (counters != nullptr) atomicAdd(&counters[id], n);
  }
  __host__ __device__ bool enabled() const { return counters != nullptr; }
};

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

struct MetricsSnapshot {
  unsigned long long c[kNumMetrics] = {};

  unsigned long long operator[](int id) const { return c[id]; }

  double hit_rate() const {
    const auto n = c[kMetricLookups];
    return n ? double(c[kMetricHits]) / double(n) : 0.0;
  }

  // What fraction of evictions CLOCK would defer. Meaningful under FIFO too —
  // that is the point of it.
  double clock_opportunity() const {
    const auto n = c[kMetricEvictCandidates];
    return n ? double(c[kMetricUsedAtPop]) / double(n) : 0.0;
  }

  // Entries examined per node actually obtained. Climbs when the ring fills
  // with stale entries or when CLOCK is deferring a lot.
  double pop_amplification() const {
    const auto got = c[kMetricEvictCandidates] - c[kMetricClaimLost];
    return got ? double(c[kMetricPopAttempts]) / double(got) : 0.0;
  }

  bool healthy() const { return c[kMetricRingExhausted] == 0ull; }
};

// Owns the device counters and a pinned staging buffer.
//
// `snapshot_async` queues a copy and returns; `read` looks at whatever the last
// completed copy left behind. Neither blocks the step. Call `read` every few
// hundred steps — the numbers are cumulative, so nothing is lost by looking
// rarely.
class MetricsBuffer {
 public:
  MetricsBuffer() {
    check(cudaMalloc(&d_, sizeof(unsigned long long) * kNumMetrics), "cudaMalloc");
    check(cudaHostAlloc(&h_, sizeof(unsigned long long) * kNumMetrics,
                        cudaHostAllocDefault),
          "cudaHostAlloc");
    for (int i = 0; i < kNumMetrics; ++i) h_[i] = 0ull;
    check(cudaMemset(d_, 0, sizeof(unsigned long long) * kNumMetrics),
          "cudaMemset");
  }
  ~MetricsBuffer() {
    cudaFree(d_);
    cudaFreeHost(h_);
  }

  MetricsBuffer(const MetricsBuffer&) = delete;
  MetricsBuffer& operator=(const MetricsBuffer&) = delete;

  MetricsView view() const { return MetricsView{d_}; }

  void snapshot_async(cudaStream_t stream) {
    check(cudaMemcpyAsync(h_, d_, sizeof(unsigned long long) * kNumMetrics,
                          cudaMemcpyDeviceToHost, stream),
          "cudaMemcpyAsync");
  }

  MetricsSnapshot read() const {
    MetricsSnapshot s;
    for (int i = 0; i < kNumMetrics; ++i) s.c[i] = h_[i];
    return s;
  }

  void reset_async(cudaStream_t stream) {
    check(cudaMemsetAsync(d_, 0, sizeof(unsigned long long) * kNumMetrics, stream),
          "cudaMemsetAsync");
  }

 private:
  static void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
      throw std::runtime_error(std::string("lw_cache metrics: ") + what + ": " +
                               cudaGetErrorString(e));
    }
  }

  unsigned long long* d_ = nullptr;
  unsigned long long* h_ = nullptr;
};

}  // namespace lw_cache
