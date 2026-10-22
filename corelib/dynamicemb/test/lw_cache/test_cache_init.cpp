// Host-side exercise of cache construction: sizing, validation, initial state.
#include <cstdio>
#include <set>
#include <vector>

#include "cache.h"

using namespace lw_cache;

static int failures = 0;
#define CHECK(cond, msg)                                             \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      ++failures;                                                    \
    }                                                                \
  } while (0)

static bool throws(const CacheConfig& cfg) {
  try {
    Cache<float> c(cfg);
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

int main() {
  // --- derived sizes ------------------------------------------------------
  {
    CacheConfig cfg;
    cfg.capacity = 100;  // deliberately not a power of two
    cfg.row_elems = 6;
    Cache<float> c(cfg);

    CHECK(c.capacity() == 100, "capacity is not rounded: the row pool is the "
                               "one array where 2x would actually cost");
    CHECK(c.num_buckets() == 256, "buckets = next_pow2(2C)");
    CHECK(c.ring_slots() == 128, "ring slots rounded up for the mask");
    CHECK(c.ring_slots() >= c.capacity(),
          "ring can always hold every node at once (INV-4)");

    CHECK(c.row_bytes() == 100u * 6u * sizeof(float), "row pool sized exactly");
    // 100*(4+8+8) + 256*4 + 128*4 = 2000 + 1024 + 512
    CHECK(c.metadata_bytes() == 3536u, "metadata accounted exactly");
  }

  // --- metadata stays a small fraction of the rows ------------------------
  {
    CacheConfig cfg;
    cfg.capacity = 1 << 20;
    cfg.row_elems = 384;  // dim 128 + Adam's two states
    Cache<float> c(cfg);
    const double ratio =
        double(c.metadata_bytes()) / double(c.row_bytes() + c.metadata_bytes());
    CHECK(ratio < 0.03, "metadata under 3% of the pool");
    std::printf("  (metadata %.2f%% of total at dim=128 fp32 Adam)\n",
                ratio * 100.0);
  }

  // --- validation ---------------------------------------------------------
  {
    CacheConfig ok;
    ok.capacity = 16;
    ok.row_elems = 4;
    CHECK(!throws(ok), "a sane config constructs");

    CacheConfig c0 = ok; c0.capacity = 0;
    CHECK(throws(c0), "zero capacity rejected");
    CacheConfig r0 = ok; r0.row_elems = 0;
    CHECK(throws(r0), "zero row width rejected");
    CacheConfig d0 = ok; d0.prefetch_depth = 0;
    CHECK(throws(d0), "zero prefetch depth rejected");

    CacheConfig clk = ok;
    clk.replacement = ReplacementPolicy::kClock;
    clk.clock_max_second_chances = 0;
    CHECK(throws(clk),
          "CLOCK without a second-chance budget rejected: an unbudgeted sweep "
          "can report the ring dry while nodes remain reclaimable");

    // capacity must cover everything an in-flight step keeps pinned
    CacheConfig tight = ok;
    tight.capacity = 100;
    tight.expected_unique_per_step = 50;
    tight.prefetch_depth = 2;
    tight.capacity_safety_factor = 1.5;  // needs 150
    CHECK(throws(tight), "capacity below the pinned working set rejected");
    tight.capacity = 200;
    CHECK(!throws(tight), "enough capacity accepted");

    // depth is what makes it bite: released nodes only return a few steps later
    CacheConfig deep = ok;
    deep.capacity = 200;
    deep.expected_unique_per_step = 50;
    deep.capacity_safety_factor = 1.5;
    deep.prefetch_depth = 1;
    CHECK(!throws(deep), "depth 1 fits");
    deep.prefetch_depth = 4;  // needs 300
    CHECK(throws(deep), "deeper pipeline needs proportionally more capacity");
  }

  // --- initial state ------------------------------------------------------
  {
    CacheConfig cfg;
    cfg.capacity = 8;
    cfg.row_elems = 2;
    Cache<float> c(cfg);

    for (StateWord s : c.download_states()) {
      CHECK(state_of(s) == kEmpty, "every node starts EMPTY");
      CHECK(ref_of(s) == 0, "unreferenced");
      CHECK(in_ring_of(s), "already listed in the ring");
      CHECK(!used_of(s), "no recency yet");
    }
    for (Key k : c.download_keys()) {
      CHECK(k == kUnboundKey, "keys start unbound");
    }
    for (int32_t h : c.download_heads()) {
      CHECK(h == kNoNode, "every bucket empty");
    }

    // the ring holds arange(capacity): every node, exactly once
    const auto ring = c.download_ring();
    uint64_t head = 0, tail = 0;
    c.download_ring_counters(&head, &tail);
    CHECK(head == 0 && tail == 8, "counters span exactly the node count");
    std::set<int32_t> listed;
    for (uint64_t p = head; p < tail; ++p) {
      listed.insert(ring[p & (c.ring_slots() - 1)]);
    }
    CHECK(listed.size() == 8, "every node listed exactly once");
    CHECK(*listed.begin() == 0 && *listed.rbegin() == 7, "arange(capacity)");
  }

  // --- reset returns to the initial state ---------------------------------
  {
    CacheConfig cfg;
    cfg.capacity = 4;
    cfg.row_elems = 2;
    Cache<float> c(cfg);

    // Scribble over the state the way a running cache would, then reset.
    auto v = c.view();
    v.chain.keys[0] = 12345;
    chain_insert(v.chain, 0);
    v.states[0] = make_state(kResident, 3u, false, true);

    c.reset();
    const auto states = c.download_states();
    CHECK(states[0] == initial_state(), "node state restored");
    CHECK(c.download_heads()[v.chain.bucket_of(12345)] == kNoNode,
          "chains cleared");
    uint64_t head = 0, tail = 0;
    c.download_ring_counters(&head, &tail);
    CHECK(head == 0 && tail == 4, "ring counters restored");
  }

  // --- CLOCK allocates its retry buffer, FIFO does not --------------------
  {
    CacheConfig fifo;
    fifo.capacity = 8;
    fifo.row_elems = 2;
    Cache<float> a(fifo);
    CHECK(!a.clock_enabled(), "FIFO by default");
    CHECK(a.retry_buffer().nodes == nullptr,
          "no retry buffer allocated when it cannot be used");

    CacheConfig clk = fifo;
    clk.replacement = ReplacementPolicy::kClock;
    Cache<float> b(clk);
    CHECK(b.clock_enabled(), "CLOCK enabled");
    CHECK(b.retry_buffer().nodes != nullptr && b.retry_buffer().capacity == 8,
          "retry buffer sized to the node count");
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
