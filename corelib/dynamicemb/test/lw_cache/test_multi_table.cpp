// Host-side exercise of multi-table dispatch.
//
// Covers: element -> table resolution including edges, per-table isolation,
// differing row widths, and the boundary-straddling fallback (simulated by
// stepping threadIdx through one block).
#include <cstdio>
#include <map>
#include <vector>

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

using Backing = StubBacking<float>;
constexpr int kLanes = 1;

struct Init {
  float operator()(Key key, int elem) const {
    return float(key) * 100.0f + float(elem);
  }
};

// The shim runs one thread per call; step threadIdx by hand to simulate a block.
struct BlockScope {
  BlockScope(unsigned threads) { blockDim = dim3(threads, 1, 1); }
  ~BlockScope() {
    blockDim = dim3(1, 1, 1);
    threadIdx = dim3(0, 0, 0);
  }
};

static std::vector<CacheConfig> configs(const std::vector<int64_t>& capacities,
                                        const std::vector<int>& row_elems) {
  std::vector<CacheConfig> out;
  for (size_t i = 0; i < capacities.size(); ++i) {
    CacheConfig c;
    c.capacity = capacities[i];
    c.row_elems = row_elems[i];
    out.push_back(c);
  }
  return out;
}

int main() {
  // --- find_table ---------------------------------------------------------
  {
    // three tables holding 2, 0 and 3 elements: the empty one is the case that
    // an off-by-one in the search gets wrong
    const int64_t range[] = {0, 2, 2, 5};
    CHECK(find_table(range, 3, 0) == 0, "first element of table 0");
    CHECK(find_table(range, 3, 1) == 0, "last element of table 0");
    CHECK(find_table(range, 3, 2) == 2, "an empty table is skipped over");
    CHECK(find_table(range, 3, 4) == 2, "last element overall");

    const int64_t one[] = {0, 7};
    for (int64_t i = 0; i < 7; ++i) {
      CHECK(find_table(one, 1, i) == 0, "single table resolves everything to 0");
    }
  }

  // --- construction: per-table sizing, shared policy -----------------------
  {
    MultiCache<float> mc(configs({16, 32, 8}, {4, 2, 6}));
    CHECK(mc.num_tables() == 3, "three tables");
    CHECK(mc.table(0).row_elems() == 4 && mc.table(1).row_elems() == 2 &&
              mc.table(2).row_elems() == 6,
          "row width is per table: a narrow table does not pay for a wide one");
    CHECK(mc.row_bytes() == (16 * 4 + 32 * 2 + 8 * 6) * sizeof(float),
          "row pools sized independently");

    auto mixed = configs({8, 8}, {2, 2});
    mixed[1].replacement = ReplacementPolicy::kClock;
    bool threw = false;
    try {
      MultiCache<float> bad(mixed);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw, "mixing replacement policies across tables is rejected");
  }

  // --- dispatch sends each key to its own table ---------------------------
  {
    MultiCache<float> mc(configs({8, 8}, {4, 4}));
    Backing backing(256, 4);
    auto mv = mc.view();

    // table 0 owns keys[0..2], table 1 owns keys[3..5]
    const std::vector<Key> keys = {11, 12, 13, 21, 22, 23};
    const std::vector<int64_t> range = {0, 3, 6};
    std::vector<int32_t> nodes(keys.size(), kNoNode);

    mt_lookup_pin_kernel(mv, keys.data(), range.data(), int64_t(keys.size()),
                         nodes.data());
    for (int32_t n : nodes) CHECK(n == kNoNode, "cold start misses everywhere");

    // allocate, tagging each slot with its table
    std::vector<AllocSlot> slots(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) {
      slots[i].key = keys[i];
      slots[i].table = int32_t(find_table(range.data(), 2, int64_t(i)));
    }
    std::vector<uint64_t> tails(2);
    for (int t = 0; t < 2; ++t) tails[t] = ring_snapshot_tail(mv.table(t).ring);

    // Two tables sharing one backing: legal when the row widths match, and
    // exercises MultiBackingView without needing two allocations.
    std::vector<Backing::View> bviews(2, backing.view());
    MultiBackingView<Backing::View> bv{bviews.data()};
    mt_claim_kernel(mv, bv, slots.data(), int64_t(slots.size()), tails.data(),
                    PopPolicy::fifo(), false);
    mt_unlink_kernel(mv, bv, slots.data(), int64_t(slots.size()));
    mt_bind_kernel(mv, slots.data(), int64_t(slots.size()));
    mt_writeback_kernel<kLanes>(mv, bv, slots.data(), int64_t(slots.size()));
    mt_fill_kernel<kLanes>(mv, bv, slots.data(), int64_t(slots.size()), Init{});

    for (const auto& s : slots) CHECK(s.allocated(), "every key got a node");

    // Each table's chain must contain only its own keys. Node ids repeat across
    // tables, which is exactly why they are meaningless without a table id.
    for (size_t i = 0; i < keys.size(); ++i) {
      const int t = int(slots[i].table);
      CHECK(chain_lookup(mv.table(t).chain, keys[i]) == slots[i].node,
            "key resolves inside its own table");
      const int other = 1 - t;
      CHECK(chain_lookup(mv.table(other).chain, keys[i]) == kNoNode,
            "and is invisible in the other table");
    }

    // a second lookup now hits, still routed correctly
    mt_lookup_pin_kernel(mv, keys.data(), range.data(), int64_t(keys.size()),
                         nodes.data());
    for (size_t i = 0; i < keys.size(); ++i) {
      CHECK(nodes[i] == slots[i].node, "hit returns the same node as allocated");
    }
    mt_release_kernel(mv, nodes.data(), range.data(), int64_t(nodes.size()));
    // release once more for the pins taken during allocate
    std::vector<int32_t> alloc_nodes(keys.size());
    for (size_t i = 0; i < keys.size(); ++i) alloc_nodes[i] = slots[i].node;
    mt_release_kernel(mv, alloc_nodes.data(), range.data(),
                      int64_t(alloc_nodes.size()));
  }

  // --- the boundary-straddling block takes the per-thread path ------------
  //
  // Under the shim every "block" is one element, so the fast path always wins
  // and the fallback never runs. Simulate a real block by stepping threadIdx,
  // with the work sized to one grid-stride iteration so the shared value stays
  // meaningful across calls.
  {
    MultiCache<float> mc(configs({8, 8}, {4, 4}));
    auto mv = mc.view();

    const std::vector<Key> keys = {11, 12, 21, 22};
    const std::vector<int64_t> range = {0, 2, 4};  // boundary in mid-block
    std::vector<int32_t> nodes(keys.size(), kNoNode);
    std::vector<int> seen_table(keys.size(), -1);

    {
      BlockScope block(4);
      for (unsigned tid = 0; tid < 4; ++tid) {
        threadIdx = dim3(tid, 1, 1);
        dispatch_by_table(mv, range.data(), int64_t(keys.size()),
                          [&](int64_t i, int t, const CacheView<float>&) {
                            seen_table[size_t(i)] = t;
                          });
      }
    }
    CHECK(seen_table[0] == 0 && seen_table[1] == 0, "first half -> table 0");
    CHECK(seen_table[2] == 1 && seen_table[3] == 1, "second half -> table 1");
    std::printf("  (straddling block resolved per-thread: %d %d %d %d)\n",
                seen_table[0], seen_table[1], seen_table[2], seen_table[3]);

    // and a block that does not straddle still resolves correctly
    const std::vector<int64_t> aligned = {0, 4, 4};
    std::fill(seen_table.begin(), seen_table.end(), -1);
    {
      BlockScope block(4);
      for (unsigned tid = 0; tid < 4; ++tid) {
        threadIdx = dim3(tid, 1, 1);
        dispatch_by_table(mv, aligned.data(), int64_t(keys.size()),
                          [&](int64_t i, int t, const CacheView<float>&) {
                            seen_table[size_t(i)] = t;
                          });
      }
    }
    for (int t : seen_table) CHECK(t == 0, "aligned block takes the fast path");
  }

  // --- tables are isolated under eviction pressure ------------------------
  //
  // Table 0 thrashes; table 1 must be unaffected. Capacity cannot be shared,
  // which is the cost of the split, and this is what that looks like.
  {
    MultiCache<float> mc(configs({2, 8}, {4, 4}));
    Backing backing(512, 4);
    auto mv = mc.view();
    std::vector<Backing::View> bviews(2, backing.view());
    MultiBackingView<Backing::View> bv{bviews.data()};
    std::map<Key, float> model;

    auto step = [&](const std::vector<Key>& keys,
                    const std::vector<int64_t>& range) {
      std::vector<int32_t> nodes(keys.size(), kNoNode);
      mt_lookup_pin_kernel(mv, keys.data(), range.data(), int64_t(keys.size()),
                           nodes.data());
      std::vector<AllocSlot> slots;
      std::vector<size_t> miss;
      for (size_t i = 0; i < keys.size(); ++i) {
        if (nodes[i] != kNoNode) continue;
        AllocSlot s;
        s.key = keys[i];
        s.table = int32_t(find_table(range.data(), 2, int64_t(i)));
        slots.push_back(s);
        miss.push_back(i);
      }
      if (!slots.empty()) {
        std::vector<uint64_t> tails(2);
        for (int t = 0; t < 2; ++t) tails[t] = ring_snapshot_tail(mv.table(t).ring);
        const int64_t m = int64_t(slots.size());
        mt_claim_kernel(mv, bv, slots.data(), m, tails.data(), PopPolicy::fifo(),
                        false);
        mt_unlink_kernel(mv, bv, slots.data(), m);
        mt_bind_kernel(mv, slots.data(), m);
        mt_writeback_kernel<kLanes>(mv, bv, slots.data(), m);
        mt_fill_kernel<kLanes>(mv, bv, slots.data(), m, Init{});
        for (size_t j = 0; j < miss.size(); ++j) nodes[miss[j]] = slots[j].node;
      }
      for (size_t i = 0; i < keys.size(); ++i) {
        CHECK(nodes[i] != kNoNode, "every key placed");
        float* row = mv.table(int(find_table(range.data(), 2, int64_t(i))))
                         .row(nodes[i]);
        for (int e = 0; e < 4; ++e) row[e] += 1.0f;
        model[keys[i]] += 1.0f;
      }
      mt_release_kernel(mv, nodes.data(), range.data(), int64_t(nodes.size()));
    };

    // table 0 cycles through 10 keys with room for 2; table 1 holds 3 steady
    for (int s = 0; s < 30; ++s) {
      std::vector<Key> keys = {Key(1000 + s % 10), Key(2001), Key(2002),
                               Key(2003)};
      step(keys, {0, 1, 4});
    }

    // flush everything
    std::vector<int64_t> node_range = {0, 2, 10};
    mt_flush_kernel<kLanes>(mv, bv, node_range.data(), 10);

    for (const auto& kv : model) {
      std::vector<float> got;
      CHECK(backing.read_row(kv.first, got), "key present after flush");
      bool ok = got.size() == 4;
      for (int e = 0; ok && e < 4; ++e) {
        ok = got[size_t(e)] == Init{}(kv.first, e) + kv.second;
      }
      CHECK(ok, "value correct across 30 steps of one-sided thrashing");
    }
    std::printf("  (table 0 thrashing at cap 2, table 1 steady: %zu keys ok)\n",
                model.size());

    unsigned long long referenced = 0;
    mt_count_referenced_kernel(mv, node_range.data(), 10, &referenced);
    CHECK(referenced == 0, "nothing left referenced across either table");
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
