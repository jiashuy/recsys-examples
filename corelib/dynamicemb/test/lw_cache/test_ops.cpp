// Host-side exercise of a full cache step, driven exactly through the phase
// order the kernels will use.
//
// Single-threaded, so what this validates is the data flow: which keys hit,
// which miss, what gets written where, and that a value survives eviction and
// reload unchanged.
#include <cstdio>
#include <map>
#include <vector>

#include "ops.h"
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

// A brand-new key's row, deterministic so the test can predict it.
struct Init {
  float operator()(Key key, int elem) const {
    return float(key) * 100.0f + float(elem);
  }
};

struct StepStats {
  int hits = 0;
  int misses = 0;
  int fresh = 0;      // backing had no record
  int loaded = 0;     // pulled over from the backing
  int bypassed = 0;   // could not be cached at all
  int wrote_back = 0;
  bool ring_exhausted = false;
};

// One step, following the phase order in ops.h. Applies `delta` to every
// element of every touched row, standing in for the optimizer.
static StepStats run_step(Cache<float>& cache, Backing& backing,
                          const std::vector<Key>& keys, float delta,
                          std::vector<int32_t>* out_nodes = nullptr) {
  StepStats st;
  auto v = cache.view();
  auto bv = backing.view();
  auto retry = cache.retry_buffer();
  RetryBuffer* retry_ptr = cache.clock_enabled() ? &retry : nullptr;

  // --- phase: lookup + pin
  std::vector<int32_t> nodes(keys.size(), kNoNode);
  std::vector<size_t> miss_idx;
  for (size_t i = 0; i < keys.size(); ++i) {
    nodes[i] = cache_lookup_pin(v, keys[i]);
    if (nodes[i] == kNoNode) miss_idx.push_back(i);
    else ++st.hits;
  }
  st.misses = int(miss_idx.size());

  // --- phase: claim
  std::vector<AllocSlot> slots(miss_idx.size());
  const uint64_t tail = ring_snapshot_tail(v.ring);
  for (size_t j = 0; j < miss_idx.size(); ++j) {
    slots[j].key = keys[miss_idx[j]];
    cache_claim(v, bv, slots[j], tail, cache.config().pop_policy(), retry_ptr);
    if (!slots[j].allocated()) {
      ++st.bypassed;
      if (row_valid(slots[j].new_row)) st.ring_exhausted = true;
    } else if (slots[j].fresh) {
      ++st.fresh;
    } else {
      ++st.loaded;
    }
  }

  // --- phase: unlink (victims leave their buckets)
  for (auto& s : slots) cache_unlink_victim(v, bv, s);

  // --- phase: bind (new keys join theirs)
  for (auto& s : slots) cache_bind(v, s);

  // --- phase: transfer
  for (auto& s : slots) {
    if (s.allocated() && row_valid(s.victim_row)) ++st.wrote_back;
    cache_writeback(v, bv, s, 0, 1);
    cache_fill(v, bv, s, Init{}, 0, 1);
    cache_publish(v, s);
  }
  for (size_t j = 0; j < miss_idx.size(); ++j) nodes[miss_idx[j]] = slots[j].node;

  // --- forward would gather here; backward updates the row in place
  for (int32_t n : nodes) {
    if (n == kNoNode) continue;
    float* row = v.row(n);
    for (int e = 0; e < kRowElems; ++e) row[e] += delta;
  }

  // --- phase: release
  for (int32_t n : nodes) cache_release(v, n);

  if (out_nodes) *out_nodes = nodes;
  return st;
}

static void flush(Cache<float>& cache, Backing& backing) {
  auto v = cache.view();
  auto bv = backing.view();
  for (int64_t n = 0; n < cache.capacity(); ++n) {
    cache_flush_node(v, bv, int32_t(n), 0, 1);
  }
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

static Cache<float> make_cache(int64_t capacity) {
  CacheConfig cfg;
  cfg.capacity = capacity;
  cfg.row_elems = kRowElems;
  return Cache<float>(cfg);
}

int main() {
  // --- cold start: everything misses, everything is fresh -----------------
  {
    Cache<float> cache = make_cache(8);
    Backing backing(64, kRowElems);

    const StepStats s = run_step(cache, backing, {1, 2, 3}, 1.0f);
    CHECK(s.hits == 0 && s.misses == 3, "cold start misses everything");
    CHECK(s.fresh == 3 && s.loaded == 0,
          "an empty backing means initialize, not transfer");
    CHECK(s.wrote_back == 0, "EMPTY nodes carry nothing to write back");
    CHECK(backing.size() == 3,
          "admission reserves a backing row up front, so eviction never has to "
          "allocate");
  }

  // --- second touch hits, and the update accumulated ----------------------
  {
    Cache<float> cache = make_cache(8);
    Backing backing(64, kRowElems);
    run_step(cache, backing, {1, 2, 3}, 1.0f);
    const StepStats s = run_step(cache, backing, {1, 2, 3}, 1.0f);
    CHECK(s.hits == 3 && s.misses == 0, "resident keys hit");
    CHECK(s.fresh == 0 && s.loaded == 0, "a hit moves nothing over PCIe");

    flush(cache, backing);
    std::vector<float> got;
    CHECK(backing.read_row(1, got), "flushed key present in the backing");
    CHECK(rows_equal(got, expect_row(1, 2.0f)),
          "both steps' updates are in the flushed row");
  }

  // --- a released node is still a hit -------------------------------------
  //
  // ref reaching zero puts the node in the ring but does not invalidate it; it
  // keeps serving until something actually pops it.
  {
    Cache<float> cache = make_cache(8);
    Backing backing(64, kRowElems);
    std::vector<int32_t> first, second;
    run_step(cache, backing, {7}, 1.0f, &first);
    const StepStats s = run_step(cache, backing, {7}, 1.0f, &second);
    CHECK(s.hits == 1, "queued node still hits");
    CHECK(first[0] == second[0], "and it is the same node");
  }

  // --- eviction round-trips the value -------------------------------------
  {
    Cache<float> cache = make_cache(2);  // deliberately tiny
    Backing backing(64, kRowElems);

    run_step(cache, backing, {10, 20}, 1.0f);   // fills the cache
    const StepStats s = run_step(cache, backing, {30, 40}, 1.0f);
    CHECK(s.misses == 2, "the new pair misses");
    CHECK(s.wrote_back == 2, "and evicts the old pair, writing both back");

    std::vector<float> got;
    CHECK(backing.read_row(10, got), "evicted key is in the backing");
    CHECK(rows_equal(got, expect_row(10, 1.0f)),
          "the evicted row carries the update it received while cached");

    // reading it back must reload rather than re-initialize
    const StepStats s2 = run_step(cache, backing, {10}, 1.0f);
    CHECK(s2.misses == 1 && s2.loaded == 1 && s2.fresh == 0,
          "a key the backing still knows is loaded, not initialized");
    flush(cache, backing);
    CHECK(backing.read_row(10, got) && rows_equal(got, expect_row(10, 2.0f)),
          "value survived eviction and reload intact (INV-7)");
  }

  // --- backing eviction: the key looks new again --------------------------
  {
    Cache<float> cache = make_cache(2);
    Backing backing(64, kRowElems);
    run_step(cache, backing, {10, 20}, 1.0f);
    run_step(cache, backing, {30, 40}, 1.0f);  // pushes 10 and 20 out to backing

    backing.evict_host({10});  // the backing forgets it, and rebuilds

    const StepStats s = run_step(cache, backing, {10}, 1.0f);
    CHECK(s.fresh == 1 && s.loaded == 0,
          "a key the backing dropped is indistinguishable from a new one, and "
          "is handled the same way");
    flush(cache, backing);
    std::vector<float> got;
    CHECK(backing.read_row(10, got) && rows_equal(got, expect_row(10, 1.0f)),
          "it starts over from the initializer");
  }

  // --- eviction still works after the backing moved every row -------------
  //
  // evict_host rebuilds, so surviving keys change row. Nothing in the cache
  // stores a row index, so nothing goes stale — this is the property that pays
  // for the extra probe on the eviction path.
  {
    Cache<float> cache = make_cache(2);
    Backing backing(64, kRowElems);
    run_step(cache, backing, {100, 200}, 5.0f);

    backing.evict_host({999});  // no such key: a pure rebuild
    const Row moved = backing.probe_host(100);

    // now evict 100 and 200 out of the cache; their rows must land correctly
    run_step(cache, backing, {300, 400}, 1.0f);
    std::vector<float> got;
    CHECK(backing.read_row(100, got), "key still resolvable after the rebuild");
    CHECK(rows_equal(got, expect_row(100, 5.0f)),
          "written to wherever the key lives now, not where it used to");
    CHECK(backing.probe_host(100) == moved, "and that is the post-rebuild row");
  }

  // --- backing full: the key is bypassed, not silently dropped ------------
  {
    Cache<float> cache = make_cache(8);
    Backing backing(4, kRowElems);  // 4 slots only

    const StepStats s = run_step(cache, backing, {1, 2, 3, 4, 5, 6}, 1.0f);
    CHECK(s.bypassed == 2, "keys past the backing's capacity are bypassed");
    CHECK(!s.ring_exhausted, "and the ring was not the reason");
    CHECK(backing.size() == 4, "backing filled exactly");
  }

  // --- ring exhausted: reported, not worked around ------------------------
  //
  // Every node pinned by one step and more keys still to place. This is a
  // capacity planning error and there is nothing to retry.
  {
    Cache<float> cache = make_cache(2);
    Backing backing(64, kRowElems);
    const StepStats s = run_step(cache, backing, {1, 2, 3, 4}, 1.0f);
    CHECK(s.bypassed == 2, "the keys that found no node");
    CHECK(s.ring_exhausted,
          "and the reason is distinguishable from a full backing");
  }

  // --- flush writes every resident row, regardless of anything ------------
  {
    Cache<float> cache = make_cache(8);
    Backing backing(64, kRowElems);
    run_step(cache, backing, {1, 2, 3}, 7.0f);

    // nothing has been evicted, so the backing rows are still untouched
    std::vector<float> got;
    CHECK(backing.read_row(1, got), "row reserved at admission");
    CHECK(!rows_equal(got, expect_row(1, 7.0f)),
          "but never written: the cache absorbed the update");

    flush(cache, backing);
    for (Key k : {1, 2, 3}) {
      CHECK(backing.read_row(k, got) && rows_equal(got, expect_row(k, 7.0f)),
            "flush wrote every resident row");
    }
  }

  // --- many steps over a working set larger than the cache ----------------
  //
  // Ends by comparing the flushed backing against a plain model of the same
  // updates. This is the P0 differential test in miniature.
  {
    Cache<float> cache = make_cache(4);
    Backing backing(256, kRowElems);
    std::map<Key, float> model;  // key -> accumulated delta

    for (int step = 0; step < 40; ++step) {
      std::vector<Key> keys;
      for (int i = 0; i < 3; ++i) keys.push_back(Key((step * 3 + i) % 11));
      // dedup, as the caller is required to
      std::vector<Key> uniq;
      for (Key k : keys) {
        bool seen = false;
        for (Key u : uniq) seen |= (u == k);
        if (!seen) uniq.push_back(k);
      }
      run_step(cache, backing, uniq, 1.0f);
      for (Key k : uniq) model[k] += 1.0f;
    }
    flush(cache, backing);

    int checked = 0;
    for (const auto& kv : model) {
      std::vector<float> got;
      CHECK(backing.read_row(kv.first, got), "every touched key is in the backing");
      CHECK(rows_equal(got, expect_row(kv.first, kv.second)),
            "cached path matches the model exactly after 40 steps of churn");
      ++checked;
    }
    CHECK(checked == 11, "the whole working set was verified");
    std::printf("  (40 steps, working set 11, capacity 4: %d keys verified)\n",
                checked);
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
