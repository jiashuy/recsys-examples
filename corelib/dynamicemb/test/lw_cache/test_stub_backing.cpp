// Host-side exercise of StubBacking's algorithmic core.
#include <cstdio>
#include <set>
#include <vector>

#include "testing/stub_backing.h"

using namespace lw_cache;
using namespace lw_cache::testing;

static int failures = 0;
#define CHECK(cond, msg)                                     \
  do {                                                       \
    if (!(cond)) {                                           \
      std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      ++failures;                                            \
    }                                                        \
  } while (0)

using Backing = StubBacking<float>;

// device view methods are host-callable under the stub
static Row dev_probe(const Backing& b, Key k) { return b.view().probe(k); }
static Row dev_insert(Backing& b, Key k) {
  auto v = b.view();
  return v.find_or_insert(k);
}

int main() {
  // --- capacity rounding -------------------------------------------------
  {
    Backing b(5, 4);
    CHECK(b.slots() == 8, "slots rounded up to power of two");
    CHECK(b.elems() == 4, "elems preserved");
  }

  // --- insert / probe agreement between host and device paths ------------
  {
    Backing b(64, 3);
    std::vector<Key> keys = {1, 2, 3, 100, 12345, -7, -99999, 1 << 20};
    for (Key k : keys) {
      const Row r = dev_insert(b, k);
      CHECK(row_valid(r), "device find_or_insert succeeded");
      CHECK(b.probe_host(k) == r, "host probe agrees with device insert row");
      CHECK(dev_probe(b, k) == r, "device probe agrees with device insert row");
    }
    CHECK(b.size() == (int64_t)keys.size(), "size counts every inserted key");

    // absent keys
    CHECK(dev_probe(b, 999999) == kRowNotFound, "absent key -> kRowNotFound");
    CHECK(!b.contains(999999), "contains() false for absent key");

    // idempotence: re-inserting returns the same row
    for (Key k : keys) {
      CHECK(dev_insert(b, k) == b.probe_host(k), "find_or_insert is idempotent");
    }
    CHECK(b.size() == (int64_t)keys.size(), "no duplicate slots consumed");
  }

  // --- rows are distinct, and payloads round-trip ------------------------
  {
    Backing b(64, 3);
    std::set<Row> seen;
    for (Key k = 0; k < 20; ++k) {
      const Row r = dev_insert(b, k);
      CHECK(seen.insert(r).second, "each key gets a distinct row");
      std::vector<float> payload = {float(k), float(k) * 2, float(k) * 3};
      CHECK(b.write_row(k, payload), "write_row found the key");
    }
    for (Key k = 0; k < 20; ++k) {
      std::vector<float> out;
      CHECK(b.read_row(k, out), "read_row found the key");
      CHECK(out.size() == 3 && out[0] == float(k) && out[2] == float(k) * 3,
            "payload round-trips");
    }
  }

  // --- table full -> kRowAllocFailed --------------------------------------
  {
    Backing b(8, 2);  // 8 slots
    int inserted = 0;
    for (Key k = 0; k < 100; ++k) {
      if (row_valid(dev_insert(b, k))) ++inserted;
    }
    CHECK(inserted == 8, "exactly `slots` keys fit");
    CHECK(!row_valid(dev_insert(b, 12345)), "full table returns kRowAllocFailed");
    CHECK(dev_insert(b, 0) == b.probe_host(0),
          "full table still resolves keys already present");
  }

  // --- eviction rebuild: payloads survive, rows move ----------------------
  {
    Backing b(64, 2);
    for (Key k = 0; k < 30; ++k) {
      dev_insert(b, k);
      b.write_row(k, {float(k), float(k) + 0.5f});
    }
    std::vector<Row> before;
    for (Key k = 0; k < 30; ++k) before.push_back(b.probe_host(k));

    b.evict_host({5, 6, 7, 8});

    CHECK(b.size() == 26, "victims removed");
    for (Key v : {5, 6, 7, 8}) {
      CHECK(!b.contains(v), "victim is gone");
      std::vector<float> out;
      CHECK(!b.read_row(v, out), "victim row unreadable");
    }
    int moved = 0;
    for (Key k = 0; k < 30; ++k) {
      if (k >= 5 && k <= 8) continue;
      std::vector<float> out;
      CHECK(b.read_row(k, out), "survivor still present");
      CHECK(out[0] == float(k) && out[1] == float(k) + 0.5f,
            "survivor payload followed its key across the rebuild");
      if (b.probe_host(k) != before[k]) ++moved;
    }
    // The point of rebuild-on-evict: rows relocate, so any cached row index
    // would now be stale.
    CHECK(moved > 0, "rebuild relocated at least one surviving key");
    std::printf("  (rebuild relocated %d/26 survivors)\n", moved);
  }

  // --- snapshot is key-ordered and complete -------------------------------
  {
    Backing b(32, 1);
    for (Key k : {9, 3, 7, 1}) {
      dev_insert(b, k);
      b.write_row(k, {float(k * 10)});
    }
    const auto snap = b.snapshot();
    CHECK(snap.size() == 4, "snapshot covers every live key");
    CHECK(snap[0].first == 1 && snap[3].first == 9, "snapshot sorted by key");
    CHECK(snap[2].second[0] == 70.f, "snapshot carries payloads");
  }

  // --- sentinel rejection --------------------------------------------------
  {
    Backing b(16, 1);
    bool threw = false;
    try {
      b.insert_host(kEmptyKey);
    } catch (const std::invalid_argument&) {
      threw = true;
    }
    CHECK(threw, "empty-slot sentinel rejected as a key");
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
