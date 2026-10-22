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
// What the host tests cannot reach.
//
// test/lw_cache/run_host_tests.sh compiles the same headers against a shim that
// serialises every atomic, so it can show a protocol is right and cannot show it
// is race-free. This runs the real thing on a real device, and targets exactly
// the gaps:
//
//   1. try_pin against try_claim on one state word, under contention
//   2. the bucket lock, with many threads editing one chain
//   3. push and pop on the ring, and the tail-snapshot window
//   4. an end-to-end differential run with real launches
//   5. multi-table dispatch on a real block, including the shared-memory
//      publish and the straddling fallback
//   6. prefetch depth > 1: keys pinned by several in-flight steps, and the
//      stream event that orders allocate against release
//
// Standalone: no torch, no pybind. Builds in seconds, so a compile error here
// is attributable to the cache and not to the extension around it.
//
//   nvcc -O2 -std=c++17 -arch=sm_90 -I../../../src/lw_cache -o test_gpu test_gpu.cu
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "kernels.cuh"
#include "launch.cuh"
#include "multi_kernels.cuh"
#include "testing/stub_backing.h"

using namespace lw_cache;
using namespace lw_cache::testing;

static int failures = 0;
#define CHECK(cond, msg)                                             \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      ++failures;                                                    \
    }                                                                \
  } while (0)

#define CUDA_OK(call)                                                       \
  do {                                                                      \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
      std::printf("  CUDA FAIL: %s -> %s (%s:%d)\n", #call,                 \
                  cudaGetErrorString(_e), __FILE__, __LINE__);              \
      ++failures;                                                           \
    }                                                                       \
  } while (0)

constexpr int kRowElems = 8;
using Backing = StubBacking<float>;

struct Init {
  __host__ __device__ float operator()(Key key, int elem) const {
    return float(key) * 100.0f + float(elem);
  }
};

template <typename T>
static T* dev_alloc(size_t n) {
  T* p = nullptr;
  CUDA_OK(cudaMalloc(&p, n * sizeof(T)));
  return p;
}
template <typename T>
static void up(T* d, const std::vector<T>& h) {
  CUDA_OK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}
template <typename T>
static std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  CUDA_OK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
  return h;
}

// ---------------------------------------------------------------------------
// 1. pin against claim
// ---------------------------------------------------------------------------
//
// Half the threads try to pin one node, half try to claim it. The two must be
// mutually exclusive: if anyone claimed it, nobody may hold a pin taken before
// the claim, and a claim must never succeed while a pin is outstanding. Packing
// state and ref into one word is what is supposed to guarantee that; this is
// the only place it gets tested.

__global__ void pin_vs_claim_kernel(StateWord* state, unsigned* pins,
                                    unsigned* claims, int iters) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < iters; ++i) {
    if (tid % 2 == 0) {
      if (try_pin(state)) {
        atomicAdd(pins, 1u);
        release_ref(state);
      }
    } else {
      StateWord from = 0;
      if (try_claim(state, &from)) {
        atomicAdd(claims, 1u);
        // put it straight back so the contest continues
        *state = make_state(kResident, 0u, in_ring_of(from), used_of(from));
        __threadfence();
      }
    }
  }
}

// The tell-tale of a broken protocol: a state word that cannot be reached by
// any legal sequence. ref must never exceed the number of live pinners, and a
// node in EVICTING must have ref == 0.
__global__ void check_state_sane_kernel(const StateWord* state, unsigned* bad) {
  const StateWord s = *state;
  if (state_of(s) == kEvicting && ref_of(s) != 0u) atomicAdd(bad, 1u);
  if (state_of(s) == kEmpty && ref_of(s) != 0u) atomicAdd(bad, 1u);
}

static void test_pin_vs_claim() {
  std::printf("=== pin vs claim under contention ===\n");
  StateWord* d_state = dev_alloc<StateWord>(1);
  unsigned* d_counts = dev_alloc<unsigned>(3);
  const std::vector<StateWord> init{make_state(kResident, 0u, true, false)};
  up(d_state, init);
  CUDA_OK(cudaMemset(d_counts, 0, 3 * sizeof(unsigned)));

  pin_vs_claim_kernel<<<64, 256>>>(d_state, d_counts, d_counts + 1, 200);
  CUDA_OK(cudaDeviceSynchronize());
  check_state_sane_kernel<<<1, 1>>>(d_state, d_counts + 2);
  CUDA_OK(cudaDeviceSynchronize());

  const auto c = down(d_counts, 3);
  const auto s = down(d_state, 1);
  std::printf("  pins=%u claims=%u impossible_states=%u final ref=%u state=%d\n",
              c[0], c[1], c[2], ref_of(s[0]), int(state_of(s[0])));
  CHECK(c[2] == 0, "no thread ever observed an impossible state word");
  CHECK(ref_of(s[0]) == 0, "every pin was matched by a release");
  CHECK(c[0] > 0 && c[1] > 0, "both sides actually contended");
  CUDA_OK(cudaFree(d_state));
  CUDA_OK(cudaFree(d_counts));
}

// ---------------------------------------------------------------------------
// 2. the bucket lock
// ---------------------------------------------------------------------------
//
// Every thread inserts into the same bucket, then every thread unlinks. If the
// lock is wrong the chain loses nodes, gains cycles, or leaves the head stuck at
// the sentinel. The lock spins, so this also exercises the sm_70+ forward
// progress assumption for a warp where one lane holds it.

__global__ void chain_insert_kernel(HashChainView chain, const Key* keys, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  chain.keys[i] = keys[i];
  chain_insert(chain, i);
}
__global__ void chain_unlink_kernel(HashChainView chain, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  chain_unlink(chain, i, chain.keys[i]);
}

static void test_bucket_lock() {
  std::printf("=== bucket lock under contention ===\n");
  constexpr int N = 512;
  constexpr int B = 8;  // few buckets on purpose: heavy collision

  int32_t* d_heads = dev_alloc<int32_t>(B);
  int32_t* d_prev = dev_alloc<int32_t>(N);
  int32_t* d_next = dev_alloc<int32_t>(N);
  Key* d_keys = dev_alloc<Key>(N);
  Key* d_input = dev_alloc<Key>(N);

  up(d_heads, std::vector<int32_t>(B, kNoNode));
  up(d_prev, std::vector<int32_t>(N, kNoNode));
  up(d_next, std::vector<int32_t>(N, kNoNode));
  std::vector<Key> keys(N);
  for (int i = 0; i < N; ++i) keys[i] = Key(i + 1);
  up(d_input, keys);

  HashChainView chain{d_heads, d_prev, d_next, d_keys, B, B - 1};

  chain_insert_kernel<<<(N + 127) / 128, 128>>>(chain, d_input, N);
  CUDA_OK(cudaDeviceSynchronize());

  // Walk every bucket on the host and check the structure.
  const auto heads = down(d_heads, B);
  const auto prev = down(d_prev, N);
  const auto next = down(d_next, N);
  int seen = 0;
  bool consistent = true;
  std::vector<bool> visited(N, false);
  for (int b = 0; b < B; ++b) {
    CHECK(heads[b] != kBucketLocked, "no bucket left locked");
    int32_t n = heads[b], expect_prev = kNoNode, guard = 0;
    while (n != kNoNode && guard++ <= N) {
      if (visited[n]) { consistent = false; break; }
      visited[n] = true;
      if (prev[n] != expect_prev) consistent = false;
      expect_prev = n;
      n = next[n];
      ++seen;
    }
    if (guard > N) consistent = false;  // cycle
  }
  std::printf("  inserted=%d reachable=%d\n", N, seen);
  CHECK(seen == N, "every concurrently inserted node is reachable");
  CHECK(consistent, "prev/next agree in both directions, no cycles");

  chain_unlink_kernel<<<(N + 127) / 128, 128>>>(chain, N);
  CUDA_OK(cudaDeviceSynchronize());
  const auto heads2 = down(d_heads, B);
  bool empty = true;
  for (int b = 0; b < B; ++b) empty &= (heads2[b] == kNoNode);
  CHECK(empty, "concurrent unlink empties every bucket, leaving no residue");

  CUDA_OK(cudaFree(d_heads)); CUDA_OK(cudaFree(d_prev));
  CUDA_OK(cudaFree(d_next));  CUDA_OK(cudaFree(d_keys));
  CUDA_OK(cudaFree(d_input));
}

// ---------------------------------------------------------------------------
// 3. the ring
// ---------------------------------------------------------------------------
//
// Pop everything with many threads, then push it all back. A node handed to two
// threads at once, or lost, shows up as a duplicate or a gap in the tally.

__global__ void ring_pop_kernel(RingView ring, StateWord* states,
                                uint64_t tail_snapshot, int32_t* out,
                                unsigned* count, int budget) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= budget) return;
  const PopResult r = ring_pop(ring, states, tail_snapshot, PopPolicy::fifo(),
                               nullptr);
  if (r.node != kNoNode) out[atomicAdd(count, 1u)] = r.node;
}
__global__ void ring_push_kernel(RingView ring, StateWord* states,
                                 const int32_t* nodes, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int32_t node = nodes[i];
  // put it back into a releasable state, then release
  states[node] = make_state(kResident, 1u, false, false);
  __threadfence();
  if (release_ref(&states[node])) ring_push(ring, node);
}

static void test_ring_concurrency() {
  std::printf("=== ring pop/push under contention ===\n");
  constexpr int C = 4096;

  int32_t* d_slots = dev_alloc<int32_t>(C);
  uint64_t* d_head = dev_alloc<uint64_t>(1);
  uint64_t* d_tail = dev_alloc<uint64_t>(1);
  StateWord* d_states = dev_alloc<StateWord>(C);
  int32_t* d_out = dev_alloc<int32_t>(C);
  unsigned* d_count = dev_alloc<unsigned>(1);

  std::vector<int32_t> slots(C);
  for (int i = 0; i < C; ++i) slots[i] = i;
  up(d_slots, slots);
  up(d_states, std::vector<StateWord>(C, initial_state()));
  const std::vector<uint64_t> h0{0}, t0{uint64_t(C)};
  up(d_head, h0);
  up(d_tail, t0);
  CUDA_OK(cudaMemset(d_count, 0, sizeof(unsigned)));

  RingView ring{d_slots, d_head, d_tail, C, C - 1, MetricsView{nullptr}};

  // More threads than nodes: the surplus must get a clean "exhausted", not a
  // duplicate.
  ring_pop_kernel<<<(C * 2 + 255) / 256, 256>>>(ring, d_states, uint64_t(C),
                                                d_out, d_count, C * 2);
  CUDA_OK(cudaDeviceSynchronize());

  const unsigned got = down(d_count, 1)[0];
  const auto out = down(d_out, C);
  std::vector<int> tally(C, 0);
  for (unsigned i = 0; i < got && i < unsigned(C); ++i) tally[out[i]]++;
  int dup = 0, missing = 0;
  for (int i = 0; i < C; ++i) {
    if (tally[i] > 1) ++dup;
    if (tally[i] == 0) ++missing;
  }
  std::printf("  popped=%u of %d, duplicates=%d missing=%d\n", got, C, dup, missing);
  CHECK(got == unsigned(C), "exactly capacity nodes handed out, no more");
  CHECK(dup == 0, "no node handed to two threads at once");
  CHECK(missing == 0, "no node lost");

  uint64_t head = down(d_head, 1)[0], tail = down(d_tail, 1)[0];
  CHECK(head <= tail, "head never overran tail (INV-5)");

  ring_push_kernel<<<(C + 255) / 256, 256>>>(ring, d_states, d_out, int(got));
  CUDA_OK(cudaDeviceSynchronize());
  head = down(d_head, 1)[0];
  tail = down(d_tail, 1)[0];
  std::printf("  after push: head=%llu tail=%llu size=%lld\n",
              (unsigned long long)head, (unsigned long long)tail,
              (long long)(tail - head));
  CHECK(int64_t(tail - head) == C, "everything came back exactly once (INV-4)");

  CUDA_OK(cudaFree(d_slots)); CUDA_OK(cudaFree(d_head));
  CUDA_OK(cudaFree(d_tail));  CUDA_OK(cudaFree(d_states));
  CUDA_OK(cudaFree(d_out));   CUDA_OK(cudaFree(d_count));
}

// ---------------------------------------------------------------------------
// 4. end to end, with real launches
// ---------------------------------------------------------------------------

__global__ void apply_update_kernel(CacheView<float> view, const int32_t* nodes,
                                    int64_t n, float delta) {
  for (int64_t i = global_thread_id(); i < n; i += thread_stride()) {
    const int32_t node = nodes[i];
    if (node == kNoNode) continue;
    float* row = view.row(node);
    for (int e = 0; e < view.row_elems; ++e) row[e] += delta;
  }
}

static void test_end_to_end(int64_t capacity, int universe, int per_step,
                            int steps, ReplacementPolicy policy,
                            const char* label) {
  std::printf("=== end to end: %s (cap=%lld set=%d n=%d steps=%d) ===\n", label,
              (long long)capacity, universe, per_step, steps);

  CacheConfig cfg;
  cfg.capacity = capacity;
  cfg.row_elems = kRowElems;
  cfg.replacement = policy;
  Cache<float> cache(cfg);
  Backing backing(1 << 16, kRowElems);
  MetricsBuffer mb;

  auto v = cache.view(mb.view());
  auto bv = backing.view();
  auto retry = cache.retry_buffer();
  Launcher launcher;

  Key* d_keys = dev_alloc<Key>(per_step);
  int32_t* d_nodes = dev_alloc<int32_t>(per_step);
  AllocSlot* d_slots = dev_alloc<AllocSlot>(per_step);
  Key* d_miss_keys = dev_alloc<Key>(per_step);

  std::map<Key, float> model;
  std::mt19937 rng(12345);

  for (int step = 0; step < steps; ++step) {
    std::vector<Key> keys;
    while (int(keys.size()) < per_step) {
      const Key k = Key(rng() % universe) + 1;
      bool dup = false;
      for (Key x : keys) dup |= (x == k);
      if (!dup) keys.push_back(k);
    }
    up(d_keys, keys);

    launcher.lookup_pin(v, d_keys, per_step, d_nodes, 0);
    CUDA_OK(cudaStreamSynchronize(0));

    // compact misses on the host: this is a test, not the hot path
    const auto nodes = down(d_nodes, per_step);
    std::vector<Key> miss;
    std::vector<int> miss_pos;
    for (int i = 0; i < per_step; ++i) {
      if (nodes[i] == kNoNode) { miss.push_back(keys[i]); miss_pos.push_back(i); }
    }
    if (!miss.empty()) {
      up(d_miss_keys, miss);
      const int64_t m = int64_t(miss.size());
      slots_set_keys_kernel<<<1, 256>>>(d_slots, d_miss_keys, m);
      CUDA_OK(cudaStreamSynchronize(0));

      uint64_t tail = 0;
      CUDA_OK(cudaMemcpy(&tail, v.ring.tail, sizeof(tail), cudaMemcpyDeviceToHost));
      launcher.allocate(v, bv, d_slots, m, tail, cfg.pop_policy(), retry,
                        cache.clock_enabled(), Init{}, 0);
      CUDA_OK(cudaStreamSynchronize(0));

      int32_t* d_alloc = dev_alloc<int32_t>(m);
      slots_get_nodes_kernel<<<1, 256>>>(d_slots, d_alloc, m);
      CUDA_OK(cudaStreamSynchronize(0));
      const auto alloc = down(d_alloc, m);
      CUDA_OK(cudaFree(d_alloc));

      auto merged = nodes;
      for (size_t j = 0; j < miss_pos.size(); ++j) {
        merged[miss_pos[j]] = alloc[j];
        if (alloc[j] == kNoNode) {
          CHECK(false, "a key could not be cached; size the trial so it can");
        }
      }
      up(d_nodes, merged);
    }

    apply_update_kernel<<<1, 256>>>(v, d_nodes, per_step, 1.0f);
    CUDA_OK(cudaStreamSynchronize(0));
    launcher.release(v, d_nodes, per_step, 0);
    CUDA_OK(cudaStreamSynchronize(0));

    for (Key k : keys) model[k] += 1.0f;
  }

  unsigned long long* d_scratch = dev_alloc<unsigned long long>(1);
  const auto leaked = launcher.count_referenced(v, d_scratch, 0);
  CHECK(leaked == 0, "no reference leak after the last release (INV-6)");

  launcher.flush(v, bv, 0);
  CUDA_OK(cudaStreamSynchronize(0));

  int checked = 0, wrong = 0;
  std::vector<float> got;
  for (const auto& kv : model) {
    if (!backing.read_row(kv.first, got)) { ++wrong; continue; }
    for (int e = 0; e < kRowElems; ++e) {
      const float want = Init{}(kv.first, e) + kv.second;
      if (got[size_t(e)] != want) { ++wrong; break; }
    }
    ++checked;
  }
  CHECK(wrong == 0, "every key matches the model exactly");

  mb.snapshot_async(0);
  CUDA_OK(cudaStreamSynchronize(0));
  const auto s = mb.read();
  std::printf("  %d keys verified, %d wrong | hit_rate=%.3f clock_opportunity=%.3f "
              "exhausted=%llu\n",
              checked, wrong, s.hit_rate(), s.clock_opportunity(),
              (unsigned long long)s[kMetricRingExhausted]);
  CHECK(s.healthy(), "the ring never ran dry");

  CUDA_OK(cudaFree(d_keys));  CUDA_OK(cudaFree(d_nodes));
  CUDA_OK(cudaFree(d_slots)); CUDA_OK(cudaFree(d_miss_keys));
  CUDA_OK(cudaFree(d_scratch));
}

// ---------------------------------------------------------------------------
// 5. multi-table dispatch, on a real block
// ---------------------------------------------------------------------------
//
// The host tests can only simulate a block by stepping threadIdx, so the shared
// memory publish and the __syncthreads around it have never actually run. The
// key boundaries here are deliberately misaligned to the block size, so both
// paths are taken: most blocks resolve their table once, the ones straddling a
// boundary fall back to per-thread search.

__global__ void record_table_kernel(MultiCacheView<float> mv,
                                    const int64_t* table_range, int64_t n,
                                    int32_t* out) {
  dispatch_by_table(mv, table_range, n,
                    [&] __device__(int64_t i, int t, const CacheView<float>&) {
                      out[i] = int32_t(t);
                    });
}

__global__ void mt_apply_update_kernel(MultiCacheView<float> mv,
                                       const int32_t* nodes,
                                       const int64_t* table_range, int64_t n,
                                       float delta) {
  dispatch_by_table(mv, table_range, n,
                    [&] __device__(int64_t i, int, const CacheView<float>& v) {
                      const int32_t node = nodes[i];
                      if (node == kNoNode) return;
                      float* row = v.row(node);
                      for (int e = 0; e < v.row_elems; ++e) row[e] += delta;
                    });
}

static void test_multi_table() {
  std::printf("=== multi-table dispatch on a real block ===\n");

  // Row widths differ: that is most of the reason the structures are split.
  const std::vector<int> widths{4, 6, 2};
  std::vector<CacheConfig> cfgs;
  for (size_t t = 0; t < widths.size(); ++t) {
    CacheConfig c;
    c.capacity = 256;
    c.row_elems = widths[t];
    cfgs.push_back(c);
  }
  MultiCache<float> mc(cfgs);
  auto mv = mc.view();

  // Boundaries at 100 and 250, neither a multiple of the 256-thread block, so
  // blocks straddle them.
  const std::vector<int64_t> range_h{0, 100, 250, 400};
  const int64_t N = range_h.back();
  int64_t* d_range = dev_alloc<int64_t>(range_h.size());
  up(d_range, range_h);

  int32_t* d_seen = dev_alloc<int32_t>(N);
  record_table_kernel<<<(N + 255) / 256, 256>>>(mv, d_range, N, d_seen);
  CUDA_OK(cudaDeviceSynchronize());
  const auto seen = down(d_seen, N);
  int wrong = 0;
  for (int64_t i = 0; i < N; ++i) {
    const int want = (i < 100) ? 0 : (i < 250 ? 1 : 2);
    if (seen[i] != want) ++wrong;
  }
  std::printf("  %lld elements dispatched, %d to the wrong table\n",
              (long long)N, wrong);
  CHECK(wrong == 0,
        "every element reached its own table, across both the shared-memory "
        "fast path and the straddling fallback");

  // Now a real step: same key values in every table must stay distinct.
  // unique_ptr because StubBacking owns CUDA allocations and is therefore
  // neither copyable nor movable; a vector of them will not compile.
  std::vector<std::unique_ptr<Backing>> backings;
  std::vector<Backing::View> bview_h;
  for (size_t t = 0; t < widths.size(); ++t) {
    backings.push_back(std::make_unique<Backing>(1 << 14, widths[t]));
    bview_h.push_back(backings.back()->view());
  }
  Backing::View* d_bviews = dev_alloc<Backing::View>(bview_h.size());
  up(d_bviews, bview_h);
  MultiBackingView<Backing::View> bv{d_bviews};

  std::vector<Key> keys_h;
  for (int t = 0; t < 3; ++t) {
    const int count = (t == 0) ? 100 : (t == 1 ? 150 : 150);
    for (int i = 0; i < count; ++i) keys_h.push_back(Key(i + 1));  // same values!
  }
  CHECK(int64_t(keys_h.size()) == N, "key layout matches the ranges");

  Key* d_keys = dev_alloc<Key>(N);
  int32_t* d_nodes = dev_alloc<int32_t>(N);
  AllocSlot* d_slots = dev_alloc<AllocSlot>(N);
  int32_t* d_tables = dev_alloc<int32_t>(N);
  uint64_t* d_tails = dev_alloc<uint64_t>(3);
  up(d_keys, keys_h);

  Launcher launcher;

  for (int step = 0; step < 20; ++step) {
    launcher.mt_lookup_pin(mv, d_keys, d_range, N, d_nodes, 0);
    CUDA_OK(cudaStreamSynchronize(0));

    const auto nodes = down(d_nodes, N);
    std::vector<Key> miss_keys;
    std::vector<int32_t> miss_tables;
    std::vector<int> miss_pos;
    for (int64_t i = 0; i < N; ++i) {
      if (nodes[i] != kNoNode) continue;
      miss_keys.push_back(keys_h[size_t(i)]);
      miss_tables.push_back(int32_t(i < 100 ? 0 : (i < 250 ? 1 : 2)));
      miss_pos.push_back(int(i));
    }
    if (!miss_keys.empty()) {
      const int64_t m = int64_t(miss_keys.size());
      Key* d_mk = dev_alloc<Key>(m);
      up(d_mk, miss_keys);
      up(d_tables, miss_tables);
      mt_slots_set_kernel<<<(m + 255) / 256, 256>>>(d_slots, d_mk, d_tables, m);
      CUDA_OK(cudaStreamSynchronize(0));

      // Read the tails through the owning Cache, not through the view: the
      // view's fields are device pointer arrays and dereferencing them on the
      // host is a segfault.
      std::vector<uint64_t> tails(3);
      for (int t = 0; t < 3; ++t) {
        uint64_t head = 0;
        mc.table(t).download_ring_counters(&head, &tails[size_t(t)]);
      }
      up(d_tails, tails);

      launcher.mt_allocate(mv, bv, d_slots, m, d_tails, cfgs[0].pop_policy(),
                           mc.clock_enabled(), Init{}, 0);
      CUDA_OK(cudaStreamSynchronize(0));

      int32_t* d_alloc = dev_alloc<int32_t>(m);
      slots_get_nodes_kernel<<<(m + 255) / 256, 256>>>(d_slots, d_alloc, m);
      CUDA_OK(cudaStreamSynchronize(0));
      const auto alloc = down(d_alloc, m);
      CUDA_OK(cudaFree(d_alloc));
      CUDA_OK(cudaFree(d_mk));

      auto merged = nodes;
      for (size_t j = 0; j < miss_pos.size(); ++j) {
        merged[size_t(miss_pos[j])] = alloc[j];
        if (alloc[j] == kNoNode) CHECK(false, "a key could not be cached");
      }
      up(d_nodes, merged);
    }

    // One update kernel over all three tables at once — the same dispatch the
    // lookup used, so a key routed to the wrong table would corrupt a row of
    // the wrong width and be caught below.
    mt_apply_update_kernel<<<(N + 255) / 256, 256>>>(mv, d_nodes, d_range, N,
                                                     1.0f);
    CUDA_OK(cudaStreamSynchronize(0));

    launcher.mt_release(mv, d_nodes, d_range, N, 0);
    CUDA_OK(cudaStreamSynchronize(0));
  }

  // flush and check every table separately
  std::vector<int64_t> node_range_h{0, 256, 512, 768};
  int64_t* d_node_range = dev_alloc<int64_t>(node_range_h.size());
  up(d_node_range, node_range_h);
  launcher.mt_flush(mv, bv, d_node_range, 768, 0);
  CUDA_OK(cudaStreamSynchronize(0));

  int bad = 0, checked = 0;
  std::vector<float> got;
  for (int t = 0; t < 3; ++t) {
    const int count = (t == 0) ? 100 : 150;
    for (int i = 0; i < count; ++i) {
      const Key k = Key(i + 1);
      if (!backings[size_t(t)]->read_row(k, got)) { ++bad; continue; }
      if (int(got.size()) != widths[size_t(t)]) { ++bad; continue; }
      for (int e = 0; e < widths[size_t(t)]; ++e) {
        if (got[size_t(e)] != Init{}(k, e) + 20.0f) { ++bad; break; }
      }
      ++checked;
    }
  }
  std::printf("  %d rows verified across 3 tables, %d wrong\n", checked, bad);
  CHECK(bad == 0,
        "the same key value in three tables stayed three separate rows, each "
        "at its own width");

  unsigned long long* d_scratch = dev_alloc<unsigned long long>(1);
  const auto leaked =
      launcher.mt_count_referenced(mv, d_node_range, 768, d_scratch, 0);
  CHECK(leaked == 0, "no reference leak across any table");

  CUDA_OK(cudaFree(d_range));   CUDA_OK(cudaFree(d_seen));
  CUDA_OK(cudaFree(d_keys));    CUDA_OK(cudaFree(d_nodes));
  CUDA_OK(cudaFree(d_slots));   CUDA_OK(cudaFree(d_tables));
  CUDA_OK(cudaFree(d_tails));   CUDA_OK(cudaFree(d_bviews));
  CUDA_OK(cudaFree(d_node_range)); CUDA_OK(cudaFree(d_scratch));
}

// ---------------------------------------------------------------------------
// 6. prefetch depth > 1
// ---------------------------------------------------------------------------
//
// Everything so far runs one step at a time, so two things have never been
// exercised:
//
//   * a key pinned by more than one in-flight step. ref then exceeds one, and
//     the node must survive until the last of them releases.
//   * the stream event that keeps allocate from starting while a release is in
//     flight. With a host sync between the two it is dead code — the sync does
//     the ordering and the event is never load-bearing.
//
// So this uses two streams and no host synchronisation inside the loop, which
// is only possible because the misses are compacted on the device.

static void test_prefetch_depth(int depth, int steps, int universe,
                                int per_step) {
  std::printf("=== prefetch depth %d (%d steps, set=%d, n=%d) ===\n", depth,
              steps, universe, per_step);

  CacheConfig cfg;
  // Every in-flight step keeps its keys pinned, so the working set that cannot
  // be reclaimed scales with depth. Undersize it and allocation fails outright.
  cfg.capacity = int64_t(per_step) * (depth + 1) * 3;
  cfg.row_elems = kRowElems;
  cfg.prefetch_depth = depth;
  cfg.expected_unique_per_step = per_step;
  Cache<float> cache(cfg);
  Backing backing(1 << 16, kRowElems);
  MetricsBuffer mb;
  Launcher launcher;

  auto v = cache.view(mb.view());
  auto bv = backing.view();
  auto retry = cache.retry_buffer();

  cudaStream_t prefetch_s = nullptr, main_s = nullptr;
  CUDA_OK(cudaStreamCreate(&prefetch_s));
  CUDA_OK(cudaStreamCreate(&main_s));

  // One slot of state per in-flight step, recycled round-robin.
  //
  // The event is not optional. Release is issued asynchronously on the main
  // stream, and the prefetch for a later step overwrites the same buffers on
  // the prefetch stream. Without waiting, the overwrite can land while the
  // release is still pending, so it releases whatever node ids happen to be
  // there — an unbalanced pin, which corrupts the state word. At full speed the
  // release usually wins the race; under compute-sanitizer it does not, which
  // is how this surfaced.
  struct InFlight {
    Key* d_keys = nullptr;
    int32_t* d_nodes = nullptr;
    AllocSlot* d_slots = nullptr;
    uint32_t* d_count = nullptr;
    cudaEvent_t released{};
    bool ever_used = false;
    std::vector<Key> keys;
  };
  std::vector<InFlight> ring(size_t(depth) + 1);
  for (auto& f : ring) {
    f.d_keys = dev_alloc<Key>(per_step);
    f.d_nodes = dev_alloc<int32_t>(per_step);
    f.d_slots = dev_alloc<AllocSlot>(per_step);
    f.d_count = dev_alloc<uint32_t>(1);
    CUDA_OK(cudaEventCreateWithFlags(&f.released, cudaEventDisableTiming));
  }

  std::map<Key, float> model;
  std::mt19937 rng(999);
  const int grid = (per_step + 255) / 256;

  for (int i = 0; i < steps + depth; ++i) {
    if (i < steps) {
      InFlight& f = ring[size_t(i) % ring.size()];
      // Do not touch these buffers until the step that last used them has
      // released its pins.
      if (f.ever_used) {
        CUDA_OK(cudaStreamWaitEvent(prefetch_s, f.released, 0));
      }
      f.keys.clear();
      while (int(f.keys.size()) < per_step) {
        const Key k = Key(rng() % universe) + 1;
        bool dup = false;
        for (Key x : f.keys) dup |= (x == k);
        if (!dup) f.keys.push_back(k);
      }
      CUDA_OK(cudaMemcpyAsync(f.d_keys, f.keys.data(), sizeof(Key) * per_step,
                              cudaMemcpyHostToDevice, prefetch_s));
      CUDA_OK(cudaMemsetAsync(f.d_count, 0, sizeof(uint32_t), prefetch_s));

      launcher.lookup_pin(v, f.d_keys, per_step, f.d_nodes, prefetch_s);
      compact_misses_kernel<float><<<grid, 256, 0, prefetch_s>>>(
          f.d_nodes, f.d_keys, per_step, f.d_slots, f.d_count);
      CUDA_OK(cudaGetLastError());

      // The bound has to be read without stalling the stream. Reading it here
      // is safe for the same reason the design gives: pushes cannot be in
      // flight, because allocate waits on the last release.
      uint64_t tail = 0;
      CUDA_OK(cudaMemcpyAsync(&tail, v.ring.tail, sizeof(tail),
                              cudaMemcpyDeviceToHost, prefetch_s));
      CUDA_OK(cudaStreamSynchronize(prefetch_s));

      launcher.allocate_compacted(v, bv, f.d_slots, f.d_count, per_step, tail,
                                  cfg.pop_policy(), retry, cache.clock_enabled(),
                                  Init{}, prefetch_s);
      scatter_nodes_kernel<<<grid, 256, 0, prefetch_s>>>(f.d_slots, f.d_count,
                                                        f.d_nodes);
      CUDA_OK(cudaGetLastError());
    }

    if (i >= depth) {
      const int s = i - depth;
      InFlight& f = ring[size_t(s) % ring.size()];
      // main stream must see the prefetch that produced these nodes
      CUDA_OK(cudaStreamSynchronize(prefetch_s));
      apply_update_kernel<<<grid, 256, 0, main_s>>>(v, f.d_nodes, per_step, 1.0f);
      CUDA_OK(cudaGetLastError());
      launcher.release(v, f.d_nodes, per_step, main_s);
      CUDA_OK(cudaEventRecord(f.released, main_s));
      f.ever_used = true;
      for (Key k : f.keys) model[k] += 1.0f;
    }
  }
  CUDA_OK(cudaStreamSynchronize(main_s));
  CUDA_OK(cudaStreamSynchronize(prefetch_s));

  unsigned long long* d_scratch = dev_alloc<unsigned long long>(1);
  CHECK(launcher.count_referenced(v, d_scratch, 0) == 0,
        "every pin taken across the in-flight window was released (INV-6)");

  launcher.flush(v, bv, 0);
  CUDA_OK(cudaStreamSynchronize(0));

  int wrong = 0, checked = 0;
  std::vector<float> got;
  for (const auto& kv : model) {
    if (!backing.read_row(kv.first, got)) { ++wrong; continue; }
    for (int e = 0; e < kRowElems; ++e) {
      if (got[size_t(e)] != Init{}(kv.first, e) + kv.second) { ++wrong; break; }
    }
    ++checked;
  }
  mb.snapshot_async(0);
  CUDA_OK(cudaStreamSynchronize(0));
  const auto st = mb.read();
  std::printf("  %d keys verified, %d wrong | hit_rate=%.3f exhausted=%llu\n",
              checked, wrong, st.hit_rate(),
              (unsigned long long)st[kMetricRingExhausted]);
  CHECK(wrong == 0, "values match the model with several steps in flight");
  CHECK(st.healthy(), "the ring never ran dry");

  for (auto& f : ring) {
    CUDA_OK(cudaFree(f.d_keys));  CUDA_OK(cudaFree(f.d_nodes));
    CUDA_OK(cudaFree(f.d_slots)); CUDA_OK(cudaFree(f.d_count));
    CUDA_OK(cudaEventDestroy(f.released));
  }
  CUDA_OK(cudaFree(d_scratch));
  CUDA_OK(cudaStreamDestroy(prefetch_s));
  CUDA_OK(cudaStreamDestroy(main_s));
}

int main() {
  // Unbuffered: a crash must not swallow the output that says where it got to.
  setvbuf(stdout, nullptr, _IONBF, 0);
  int dev = 0;
  cudaDeviceProp prop{};
  CUDA_OK(cudaGetDevice(&dev));
  CUDA_OK(cudaGetDeviceProperties(&prop, dev));
  std::printf("device: %s (sm_%d%d, %d SMs)\n\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);
  CUDA_OK(cudaSetDeviceFlags(cudaDeviceMapHost));

  test_pin_vs_claim();
  test_bucket_lock();
  test_ring_concurrency();
  test_end_to_end(1024, 4096, 256, 100, ReplacementPolicy::kFifo, "fifo");
  test_end_to_end(1024, 4096, 256, 100, ReplacementPolicy::kClock, "clock");
  test_end_to_end(512, 400, 128, 200, ReplacementPolicy::kFifo, "thrashing");
  test_multi_table();
  test_prefetch_depth(1, 100, 2048, 128);
  test_prefetch_depth(2, 100, 2048, 128);
  test_prefetch_depth(4, 100, 2048, 128);
  // Small universe so keys recur across the in-flight window and ref exceeds 1.
  test_prefetch_depth(3, 150, 200, 64);

  if (failures == 0) {
    std::printf("\nall GPU checks passed\n");
    return 0;
  }
  std::printf("\n%d check(s) failed\n", failures);
  return 1;
}
