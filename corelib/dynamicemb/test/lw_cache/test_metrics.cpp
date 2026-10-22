// Host-side exercise of the counters.
//
// Checks that each one is bumped where it should be and nowhere else, and that
// the derived ratios mean what they claim. Contention on the counters and the
// async copy-out are not covered.
#include <cstdio>
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

constexpr int kRowElems = 2;
constexpr int kLanes = 1;
using Backing = StubBacking<float>;

struct Init {
  float operator()(Key key, int elem) const {
    return float(key) * 100.0f + float(elem);
  }
};

// A step driven through the ops, with counters attached.
static void run_step(Cache<float>& cache, Backing& backing, MetricsBuffer& mb,
                     const std::vector<Key>& keys, float delta) {
  auto v = cache.view(mb.view());
  auto bv = backing.view();
  auto retry = cache.retry_buffer();
  RetryBuffer* retry_ptr = cache.clock_enabled() ? &retry : nullptr;

  std::vector<int32_t> nodes(keys.size(), kNoNode);
  std::vector<AllocSlot> slots;
  std::vector<size_t> miss;
  for (size_t i = 0; i < keys.size(); ++i) {
    nodes[i] = cache_lookup_pin(v, keys[i]);
    if (nodes[i] == kNoNode) {
      AllocSlot s;
      s.key = keys[i];
      slots.push_back(s);
      miss.push_back(i);
    }
  }
  if (!slots.empty()) {
    const uint64_t tail = ring_snapshot_tail(v.ring);
    for (auto& s : slots) {
      cache_claim(v, bv, s, tail, cache.config().pop_policy(), retry_ptr);
    }
    for (auto& s : slots) cache_unlink_victim(v, bv, s);
    for (auto& s : slots) cache_bind(v, s);
    for (auto& s : slots) {
      cache_writeback(v, bv, s, 0, kLanes);
      cache_fill(v, bv, s, Init{}, 0, kLanes);
      cache_publish(v, s);
    }
    for (size_t j = 0; j < miss.size(); ++j) nodes[miss[j]] = slots[j].node;
  }
  for (int32_t n : nodes) {
    if (n == kNoNode) continue;
    float* row = v.row(n);
    for (int e = 0; e < kRowElems; ++e) row[e] += delta;
  }
  for (int32_t n : nodes) cache_release(v, n);
}

static Cache<float> make_cache(int64_t capacity, ReplacementPolicy pol) {
  CacheConfig cfg;
  cfg.capacity = capacity;
  cfg.row_elems = kRowElems;
  cfg.replacement = pol;
  return Cache<float>(cfg);
}

int main() {
  // --- disabled by default, and free when disabled ------------------------
  {
    Cache<float> cache = make_cache(8, ReplacementPolicy::kFifo);
    Backing backing(64, kRowElems);
    CHECK(!cache.view().metrics.enabled(),
          "a view built without counters is disabled");
    // must not crash: every bump is a null check
    MetricsBuffer unused;
    (void)unused;
    auto v = cache.view();
    auto bv = backing.view();
    AllocSlot s;
    s.key = 5;
    cache_claim(v, bv, s, ring_snapshot_tail(v.ring), PopPolicy::fifo(), nullptr);
    CHECK(s.allocated(), "the disabled path still works");
  }

  // --- hits and misses ----------------------------------------------------
  {
    Cache<float> cache = make_cache(8, ReplacementPolicy::kFifo);
    Backing backing(64, kRowElems);
    MetricsBuffer mb;

    run_step(cache, backing, mb, {1, 2, 3}, 1.0f);
    mb.snapshot_async(nullptr);
    auto s = mb.read();
    CHECK(s[kMetricLookups] == 3, "one lookup per key");
    CHECK(s[kMetricHits] == 0, "cold start hits nothing");
    CHECK(s[kMetricFresh] == 3 && s[kMetricLoaded] == 0,
          "an empty backing means initialise, not transfer");
    CHECK(s[kMetricWrittenBack] == 0, "EMPTY victims carry nothing out");
    CHECK(s.hit_rate() == 0.0, "hit rate zero");

    run_step(cache, backing, mb, {1, 2, 3}, 1.0f);
    mb.snapshot_async(nullptr);
    s = mb.read();
    CHECK(s[kMetricLookups] == 6 && s[kMetricHits] == 3, "counters accumulate");
    CHECK(s.hit_rate() == 0.5, "half of all lookups have hit so far");
    CHECK(s.healthy(), "no exhaustion");
  }

  // --- eviction accounting ------------------------------------------------
  {
    Cache<float> cache = make_cache(2, ReplacementPolicy::kFifo);
    Backing backing(64, kRowElems);
    MetricsBuffer mb;

    run_step(cache, backing, mb, {10, 20}, 1.0f);  // fills
    run_step(cache, backing, mb, {30, 40}, 1.0f);  // evicts both
    mb.snapshot_async(nullptr);
    auto s = mb.read();
    CHECK(s[kMetricWrittenBack] == 2, "both victims written out");
    CHECK(s[kMetricEvictCandidates] == 4, "four nodes taken in total");

    run_step(cache, backing, mb, {10}, 1.0f);
    mb.snapshot_async(nullptr);
    s = mb.read();
    CHECK(s[kMetricLoaded] == 1,
          "a key the backing still knows is transferred, not initialised");
  }

  // --- the CLOCK opportunity indicator, measured under FIFO ---------------
  //
  // The whole point: the recency bit is set on every hit regardless of policy,
  // so a plain FIFO run already says how much CLOCK would have deferred.
  {
    Cache<float> cache = make_cache(4, ReplacementPolicy::kFifo);
    Backing backing(256, kRowElems);
    MetricsBuffer mb;

    // The bit is set by try_pin, i.e. only on a *hit* — a miss installs the
    // node with it clear. So a key has to be hit once and then left idle long
    // enough for the hand to reach it. Hitting it every step would report zero:
    // its entry would be pinned whenever the hand arrived and dropped as stale,
    // the bit never examined. FIFO already protects what is in use; this counts
    // only what CLOCK adds on top, which is keys hot recently but idle now.
    for (int round = 0; round < 6; ++round) {
      const Key hot = Key(7 + round);
      const Key base = Key(1000 + round * 10);
      run_step(cache, backing, mb, {hot, base}, 1.0f);      // miss: bit clear
      run_step(cache, backing, mb, {hot, base + 1}, 1.0f);  // hit: bit set
      for (int i = 2; i < 8; ++i) {                         // stream past it
        run_step(cache, backing, mb, {base + Key(i)}, 1.0f);
      }
    }
    mb.snapshot_async(nullptr);
    const auto s = mb.read();

    CHECK(s[kMetricEvictCandidates] > 0, "evictions happened");
    CHECK(s[kMetricUsedAtPop] > 0,
          "some of them were of nodes hit since the hand last passed — which is "
          "exactly what CLOCK would have deferred");
    CHECK(s.clock_opportunity() > 0.0 && s.clock_opportunity() <= 1.0,
          "the ratio is a fraction");
    CHECK(s[kMetricSecondChances] == 0,
          "and FIFO granted none of them, which is why the number is an "
          "opportunity rather than a report");
    std::printf("  (FIFO run: clock_opportunity=%.3f over %llu evictions)\n",
                s.clock_opportunity(),
                (unsigned long long)s[kMetricEvictCandidates]);
  }

  // --- with CLOCK on, the deferrals show up -------------------------------
  {
    Cache<float> cache = make_cache(4, ReplacementPolicy::kClock);
    Backing backing(256, kRowElems);
    MetricsBuffer mb;
    for (int round = 0; round < 6; ++round) {
      const Key hot = Key(7 + round);
      const Key base = Key(1000 + round * 10);
      run_step(cache, backing, mb, {hot, base}, 1.0f);
      run_step(cache, backing, mb, {hot, base + 1}, 1.0f);
      for (int i = 2; i < 8; ++i) {
        run_step(cache, backing, mb, {base + Key(i)}, 1.0f);
      }
    }
    mb.snapshot_async(nullptr);
    const auto s = mb.read();
    CHECK(s[kMetricSecondChances] > 0, "CLOCK actually deferred evictions");
    CHECK(s.pop_amplification() >= 1.0,
          "and paid for it with extra entries examined per node obtained");
    std::printf("  (CLOCK run: %llu second chances, pop amplification %.2f)\n",
                (unsigned long long)s[kMetricSecondChances],
                s.pop_amplification());
  }

  // --- stale entries ------------------------------------------------------
  {
    Cache<float> cache = make_cache(4, ReplacementPolicy::kFifo);
    Backing backing(64, kRowElems);
    MetricsBuffer mb;
    auto v = cache.view(mb.view());

    run_step(cache, backing, mb, {1, 2, 3, 4}, 1.0f);
    // pin node 0 so its queued entry goes stale, then drain ahead of it
    CHECK(try_pin(&v.states[0]), "hold a queued node");
    for (int i = 0; i < 3; ++i) {
      (void)ring_pop(v.ring, v.states, ring_snapshot_tail(v.ring),
                     PopPolicy::fifo(), nullptr);
    }
    (void)ring_pop(v.ring, v.states, ring_snapshot_tail(v.ring),
                   PopPolicy::fifo(), nullptr);
    mb.snapshot_async(nullptr);
    const auto s = mb.read();
    CHECK(s[kMetricStaleDropped] >= 1, "the stale entry was counted as dropped");
    CHECK(s[kMetricRingExhausted] >= 1,
          "and running out was counted separately from it");
    CHECK(!s.healthy(), "exhaustion is what healthy() reports on");
  }

  // --- backing full -------------------------------------------------------
  {
    Cache<float> cache = make_cache(16, ReplacementPolicy::kFifo);
    Backing backing(4, kRowElems);
    MetricsBuffer mb;
    run_step(cache, backing, mb, {1, 2, 3, 4, 5, 6}, 1.0f);
    mb.snapshot_async(nullptr);
    const auto s = mb.read();
    CHECK(s[kMetricBackingFull] == 2, "the two keys with nowhere to live");
    CHECK(s[kMetricRingExhausted] == 0,
          "and the ring was not the reason, which the counters keep apart");
  }

  // --- reset --------------------------------------------------------------
  {
    Cache<float> cache = make_cache(8, ReplacementPolicy::kFifo);
    Backing backing(64, kRowElems);
    MetricsBuffer mb;
    run_step(cache, backing, mb, {1, 2}, 1.0f);
    mb.reset_async(nullptr);
    mb.snapshot_async(nullptr);
    const auto s = mb.read();
    for (int i = 0; i < kNumMetrics; ++i) {
      CHECK(s[i] == 0, "reset cleared every counter");
    }
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
