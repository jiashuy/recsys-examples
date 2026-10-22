// Host-side exercise of the bucket chains.
//
// Single-threaded, so this validates chain structure — lookup, head/middle/tail
// removal, collisions — and not the bucket lock under contention.
#include <cstdio>
#include <set>
#include <vector>

#include "hash_chain.h"

using namespace lw_cache;

static int failures = 0;
#define CHECK(cond, msg)                                             \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      ++failures;                                                    \
    }                                                                \
  } while (0)

struct Fixture {
  std::vector<int32_t> heads;
  std::vector<int32_t> prev, next;
  std::vector<Key> keys;
  int64_t buckets;

  Fixture(int64_t capacity, int64_t num_buckets) : buckets(num_buckets) {
    heads.assign(static_cast<size_t>(num_buckets), kNoNode);
    prev.assign(static_cast<size_t>(capacity), kNoNode);
    next.assign(static_cast<size_t>(capacity), kNoNode);
    keys.assign(static_cast<size_t>(capacity), 0);
  }

  HashChainView view() {
    return HashChainView{heads.data(), prev.data(),  next.data(),
                         keys.data(),  buckets,      buckets - 1};
  }

  void bind(int32_t node, Key k) {
    keys[static_cast<size_t>(node)] = k;
    auto v = view();
    chain_insert(v, node);
  }
  void unbind(int32_t node) {
    auto v = view();
    chain_unlink(v, node, keys[static_cast<size_t>(node)]);
  }
  int32_t find(Key k) {
    auto v = view();
    return chain_lookup(v, k);
  }

  // Walk a bucket, checking the links agree in both directions.
  std::vector<int32_t> walk(int64_t b, bool* consistent) {
    std::vector<int32_t> out;
    *consistent = true;
    int32_t n = heads[static_cast<size_t>(b)];
    int32_t expect_prev = kNoNode;
    int guard = 0;
    while (n != kNoNode) {
      if (prev[static_cast<size_t>(n)] != expect_prev) *consistent = false;
      out.push_back(n);
      expect_prev = n;
      n = next[static_cast<size_t>(n)];
      if (++guard > 10000) {  // a cycle would otherwise hang the test
        *consistent = false;
        break;
      }
    }
    return out;
  }

  bool all_buckets_consistent() {
    for (int64_t b = 0; b < buckets; ++b) {
      bool ok = false;
      walk(b, &ok);
      if (!ok) return false;
    }
    return true;
  }
};

// Keys that share a bucket, found by construction rather than luck.
static std::vector<Key> colliding_keys(const HashChainView& v, int count) {
  std::vector<Key> out;
  const int64_t target = v.bucket_of(1);
  for (Key k = 1; out.size() < static_cast<size_t>(count) && k < 2000000; ++k) {
    if (v.bucket_of(k) == target) out.push_back(k);
  }
  return out;
}

int main() {
  // --- empty table --------------------------------------------------------
  {
    Fixture f(8, 8);
    CHECK(f.find(42) == kNoNode, "lookup in an empty table misses");
  }

  // --- insert / lookup / unlink, no collisions ----------------------------
  {
    Fixture f(64, 64);
    for (int32_t n = 0; n < 20; ++n) f.bind(n, Key(1000 + n));
    for (int32_t n = 0; n < 20; ++n) {
      CHECK(f.find(Key(1000 + n)) == n, "every inserted key is found");
    }
    CHECK(f.find(999) == kNoNode, "absent key misses");
    CHECK(f.all_buckets_consistent(), "prev/next agree everywhere");

    for (int32_t n = 0; n < 20; ++n) {
      f.unbind(n);
      CHECK(f.find(Key(1000 + n)) == kNoNode, "unlinked key no longer found");
      CHECK(f.all_buckets_consistent(), "chains stay consistent through removal");
    }
    for (int64_t b = 0; b < f.buckets; ++b) {
      CHECK(f.heads[static_cast<size_t>(b)] == kNoNode, "all buckets empty again");
    }
  }

  // --- a single bucket holding a long chain -------------------------------
  {
    Fixture f(64, 16);
    const auto ks = colliding_keys(f.view(), 6);
    CHECK(ks.size() == 6, "found six keys sharing a bucket");
    const int64_t b = f.view().bucket_of(ks[0]);

    for (size_t i = 0; i < ks.size(); ++i) f.bind(int32_t(i), ks[i]);
    bool ok = false;
    auto chain = f.walk(b, &ok);
    CHECK(ok, "chain links agree in both directions");
    CHECK(chain.size() == 6, "all six landed in one bucket");
    CHECK(chain[0] == 5, "head insertion puts the newest first");
    for (size_t i = 0; i < ks.size(); ++i) {
      CHECK(f.find(ks[i]) == int32_t(i), "each colliding key resolves to its node");
    }

    // remove from the middle
    f.unbind(3);
    chain = f.walk(b, &ok);
    CHECK(ok && chain.size() == 5, "middle removal keeps the chain intact");
    CHECK(f.find(ks[3]) == kNoNode, "removed key gone");
    CHECK(f.find(ks[2]) == 2 && f.find(ks[4]) == 4, "neighbours still reachable");

    // remove the head
    const int32_t head = chain[0];
    f.unbind(head);
    chain = f.walk(b, &ok);
    CHECK(ok && chain.size() == 4, "head removal keeps the chain intact");
    CHECK(f.heads[static_cast<size_t>(b)] != head, "bucket head advanced");

    // remove the tail
    chain = f.walk(b, &ok);
    const int32_t tail = chain.back();
    f.unbind(tail);
    chain = f.walk(b, &ok);
    CHECK(ok && chain.size() == 3, "tail removal keeps the chain intact");

    // drain the rest
    while (!chain.empty()) {
      f.unbind(chain[0]);
      chain = f.walk(b, &ok);
      CHECK(ok, "chain consistent while draining");
    }
    CHECK(f.heads[static_cast<size_t>(b)] == kNoNode, "bucket empty");
  }

  // --- node reuse: unlink from the old bucket, insert into the new ---------
  //
  // This is what allocate does across passes 4a and 4b, and the reason unlink
  // takes the outgoing key explicitly: by 4b the node's key field already holds
  // the incoming one.
  {
    Fixture f(16, 16);
    f.bind(0, 100);
    f.bind(1, 200);
    const Key old_key = f.keys[0];

    auto v = f.view();
    chain_unlink(v, 0, old_key);       // pass 4a
    f.keys[0] = 300;                   // rebind
    chain_insert(v, 0);                // pass 4b

    CHECK(f.find(100) == kNoNode, "old key no longer resolves");
    CHECK(f.find(300) == 0, "new key resolves to the reused node");
    CHECK(f.find(200) == 1, "the untouched node is unaffected");
    CHECK(f.all_buckets_consistent(), "chains consistent after the swap");
  }

  // --- churn: repeated bind/unbind leaves no residue ----------------------
  {
    Fixture f(32, 16);
    for (int round = 0; round < 50; ++round) {
      for (int32_t n = 0; n < 32; ++n) f.bind(n, Key(round * 1000 + n));
      CHECK(f.all_buckets_consistent(), "consistent after a full round of inserts");
      for (int32_t n = 0; n < 32; ++n) f.unbind(n);
      CHECK(f.all_buckets_consistent(), "consistent after a full round of removals");
    }
    for (int64_t b = 0; b < f.buckets; ++b) {
      CHECK(f.heads[static_cast<size_t>(b)] == kNoNode, "no residue after churn");
    }
    for (int32_t n = 0; n < 32; ++n) {
      CHECK(f.prev[static_cast<size_t>(n)] == kNoNode &&
                f.next[static_cast<size_t>(n)] == kNoNode,
            "unlinked nodes hold no dangling pointers");
    }
  }

  // --- the lock is released, and a locked bucket is never walked ----------
  {
    Fixture f(16, 16);
    f.bind(0, 55);
    for (int64_t b = 0; b < f.buckets; ++b) {
      CHECK(f.heads[static_cast<size_t>(b)] != kBucketLocked,
            "no bucket left locked after an operation");
    }
    auto v = f.view();
    const int64_t b = v.bucket_of(55);
    const int32_t saved = lock_bucket(v, b);
    CHECK(f.heads[static_cast<size_t>(b)] == kBucketLocked, "lock marks the head");
    CHECK(chain_lookup(v, 55) == kNoNode,
          "a lookup that races a locked bucket misses instead of walking a sentinel");
    unlock_bucket(v, b, saved);
    CHECK(f.find(55) == 0, "lookup works again once unlocked");
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
