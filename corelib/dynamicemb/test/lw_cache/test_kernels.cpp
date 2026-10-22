// Host-side exercise of the kernel bodies.
//
// Under the shim the launch geometry is one thread in one block, and every
// kernel is a grid-stride loop, so a direct call still visits every element.
// What this checks is that the kernels wire the ops together correctly and
// agree with the per-key path; the launch configuration and all concurrency
// remain unverified.
#include <cstdio>
#include <map>
#include <vector>

#include "kernels.cuh"
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

constexpr int kRowElems = 4;
using Backing = StubBacking<float>;

struct Init {
  float operator()(Key key, int elem) const {
    return float(key) * 100.0f + float(elem);
  }
};

// One lane per row: the shim has no warps.
constexpr int kLanes = 1;

// A step driven entirely through the kernels, in the documented launch order.
static std::vector<int32_t> run_step_kernels(Cache<float>& cache,
                                             Backing& backing,
                                             const std::vector<Key>& keys,
                                             float delta) {
  auto v = cache.view();
  auto bv = backing.view();
  const int64_t n = static_cast<int64_t>(keys.size());

  std::vector<int32_t> nodes(keys.size(), kNoNode);
  lookup_pin_kernel(v, keys.data(), n, nodes.data());

  // compact the misses
  std::vector<AllocSlot> slots;
  std::vector<size_t> miss_idx;
  for (size_t i = 0; i < keys.size(); ++i) {
    if (nodes[i] == kNoNode) {
      AllocSlot s;
      s.key = keys[i];
      slots.push_back(s);
      miss_idx.push_back(i);
    }
  }
  const int64_t m = static_cast<int64_t>(slots.size());

  if (m > 0) {
    const uint64_t tail = ring_snapshot_tail(v.ring);
    auto retry = cache.retry_buffer();
    claim_kernel(v, bv, slots.data(), m, tail, cache.config().pop_policy(), retry,
                 cache.clock_enabled());
    if (cache.clock_enabled()) {
      absorb_retry_kernel(v, retry);
      // The launcher clears the count with a stream-ordered memset; mirror that
      // rather than clearing it here. An earlier version of this test reset it
      // itself, which patched over the fact that nothing else did — the count
      // then carried into the next step and made absorb re-push nodes that were
      // already queued. It only failed on a real device, under real
      // concurrency.
      *retry.count = 0;
    }
    unlink_kernel(v, bv, slots.data(), m);
    bind_kernel(v, slots.data(), m);
    writeback_kernel<kLanes>(v, bv, slots.data(), m);
    fill_kernel<kLanes>(v, bv, slots.data(), m, Init{});
    for (size_t j = 0; j < miss_idx.size(); ++j) {
      nodes[miss_idx[j]] = slots[j].node;
    }
  }

  // forward + backward
  for (int32_t node : nodes) {
    if (node == kNoNode) continue;
    float* row = v.row(node);
    for (int e = 0; e < kRowElems; ++e) row[e] += delta;
  }

  release_kernel(v, nodes.data(), n);
  return nodes;
}

static std::vector<float> expect_row(Key key, float total_delta) {
  std::vector<float> r(kRowElems);
  Init init;
  for (int e = 0; e < kRowElems; ++e) r[e] = init(key, e) + total_delta;
  return r;
}

static bool rows_equal(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

static Cache<float> make_cache(int64_t capacity, ReplacementPolicy pol) {
  CacheConfig cfg;
  cfg.capacity = capacity;
  cfg.row_elems = kRowElems;
  cfg.replacement = pol;
  return Cache<float>(cfg);
}

int main() {
  // --- a step through the kernels behaves like the per-key path -----------
  {
    Cache<float> cache = make_cache(8, ReplacementPolicy::kFifo);
    Backing backing(64, kRowElems);

    auto n1 = run_step_kernels(cache, backing, {1, 2, 3}, 1.0f);
    for (int32_t x : n1) CHECK(x != kNoNode, "cold start allocated every key");

    auto n2 = run_step_kernels(cache, backing, {1, 2, 3}, 1.0f);
    CHECK(n1 == n2, "the second step hits the same nodes");

    flush_kernel<kLanes>(cache.view(), backing.view());
    std::vector<float> got;
    for (Key k : {1, 2, 3}) {
      CHECK(backing.read_row(k, got) && rows_equal(got, expect_row(k, 2.0f)),
            "both steps' updates reached the backing");
    }
  }

  // --- grid-stride: one thread really does cover every element ------------
  {
    Cache<float> cache = make_cache(64, ReplacementPolicy::kFifo);
    Backing backing(256, kRowElems);
    std::vector<Key> keys;
    for (Key k = 0; k < 50; ++k) keys.push_back(k + 1);

    auto nodes = run_step_kernels(cache, backing, keys, 3.0f);
    int allocated = 0;
    for (int32_t x : nodes) allocated += (x != kNoNode);
    CHECK(allocated == 50, "all fifty keys handled by a single-thread launch");

    flush_kernel<kLanes>(cache.view(), backing.view());
    std::vector<float> got;
    CHECK(backing.read_row(50, got) && rows_equal(got, expect_row(50, 3.0f)),
          "the last element was not skipped");
  }

  // --- eviction through the kernels ---------------------------------------
  {
    Cache<float> cache = make_cache(2, ReplacementPolicy::kFifo);
    Backing backing(64, kRowElems);
    run_step_kernels(cache, backing, {10, 20}, 4.0f);
    run_step_kernels(cache, backing, {30, 40}, 1.0f);

    std::vector<float> got;
    CHECK(backing.read_row(10, got) && rows_equal(got, expect_row(10, 4.0f)),
          "writeback_kernel wrote the evicted row");
  }

  // --- CLOCK: the retry buffer is drained back into the ring --------------
  {
    Cache<float> cache = make_cache(4, ReplacementPolicy::kClock);
    Backing backing(64, kRowElems);
    run_step_kernels(cache, backing, {1, 2, 3, 4}, 1.0f);
    // touch one of them so it carries the recency bit into the next allocate
    run_step_kernels(cache, backing, {1}, 1.0f);

    uint64_t head_before = 0, tail_before = 0;
    cache.download_ring_counters(&head_before, &tail_before);
    run_step_kernels(cache, backing, {5}, 1.0f);
    uint64_t head_after = 0, tail_after = 0;
    cache.download_ring_counters(&head_after, &tail_after);

    CHECK(tail_after >= tail_before,
          "absorb_retry_kernel appended spared nodes rather than losing them");
    CHECK(*cache.retry_buffer().count == 0,
          "retry buffer reset; a count that survives makes the next step "
          "re-push nodes that are already in the ring");

    // the whole set must still be accounted for
    flush_kernel<kLanes>(cache.view(), backing.view());
    std::vector<float> got;
    CHECK(backing.read_row(1, got), "the spared hot key is still tracked");
  }

  // --- the reference sweep sees an unbalanced release ---------------------
  {
    Cache<float> cache = make_cache(4, ReplacementPolicy::kFifo);
    Backing backing(64, kRowElems);
    run_step_kernels(cache, backing, {1, 2}, 1.0f);

    unsigned long long referenced = 0;
    count_referenced_kernel(cache.view(), &referenced);
    CHECK(referenced == 0,
          "a completed step leaves nothing referenced; this is the free INV-6 "
          "check a checkpoint gets");

    // simulate a leaked pin: one step forgets to release
    auto v = cache.view();
    CHECK(try_pin(&v.states[0]), "pin without a matching release");
    referenced = 0;
    count_referenced_kernel(v, &referenced);
    CHECK(referenced == 1, "the sweep finds the leak");
  }

  // --- forty steps of churn, compared against a plain model ---------------
  {
    Cache<float> cache = make_cache(4, ReplacementPolicy::kFifo);
    Backing backing(256, kRowElems);
    std::map<Key, float> model;

    for (int step = 0; step < 40; ++step) {
      std::vector<Key> uniq;
      for (int i = 0; i < 3; ++i) {
        const Key k = Key((step * 3 + i) % 11) + 1;
        bool seen = false;
        for (Key u : uniq) seen |= (u == k);
        if (!seen) uniq.push_back(k);
      }
      run_step_kernels(cache, backing, uniq, 1.0f);
      for (Key k : uniq) model[k] += 1.0f;
    }
    flush_kernel<kLanes>(cache.view(), backing.view());

    for (const auto& kv : model) {
      std::vector<float> got;
      CHECK(backing.read_row(kv.first, got) &&
                rows_equal(got, expect_row(kv.first, kv.second)),
            "kernel path matches the model after 40 steps of eviction churn");
    }
    std::printf("  (40 steps through the kernels, %zu keys verified)\n",
                model.size());
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
