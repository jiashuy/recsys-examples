// Host-side exercise of the reclaim ring.
//
// Single-threaded: this validates the pop/push protocol, not its behaviour
// under contention.
#include <cstdio>
#include <vector>

#include "ring.h"

using namespace lw_cache;

static int failures = 0;
#define CHECK(cond, msg)                                             \
  do {                                                               \
    if (!(cond)) {                                                   \
      std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__); \
      ++failures;                                                    \
    }                                                                \
  } while (0)

// A ring plus the node states it indexes, wired the way the cache will.
struct Fixture {
  std::vector<int32_t> slots;
  std::vector<StateWord> states;
  uint64_t head = 0, tail = 0;
  int64_t cap;

  explicit Fixture(int64_t capacity) : cap(capacity) {
    slots.resize(static_cast<size_t>(capacity));
    states.assign(static_cast<size_t>(capacity), initial_state());
    for (int64_t i = 0; i < capacity; ++i) slots[static_cast<size_t>(i)] = int32_t(i);
    head = 0;
    tail = static_cast<uint64_t>(capacity);  // arange(C): every node enqueued
  }

  RingView view() { return RingView{slots.data(), &head, &tail, cap, cap - 1}; }
  int64_t live() const { return static_cast<int64_t>(tail - head); }

  PopResult pop(const PopPolicy& p = PopPolicy::fifo(), RetryBuffer* r = nullptr) {
    return ring_pop(view(), states.data(), tail, p, r);
  }
  // Simulate a node being taken, used for a step, then released.
  void make_resident(int32_t n) {
    states[static_cast<size_t>(n)] = loading_state();
    mark_resident(&states[static_cast<size_t>(n)]);
  }
  void release(int32_t n) {
    if (release_ref(&states[static_cast<size_t>(n)])) ring_push(view(), n);
  }
};

int main() {
  // --- arange init: every node starts reclaimable -------------------------
  {
    Fixture f(8);
    CHECK(f.live() == 8, "ring starts full");
    for (int i = 0; i < 8; ++i) {
      const PopResult r = f.pop();
      CHECK(r.node == i, "pops in arange order");
      CHECK(!needs_writeback(r.claimed_from), "EMPTY node needs no writeback");
    }
    CHECK(f.pop().node == kNoNode, "ring exhausted after C pops");
    CHECK(f.head <= f.tail, "exhausted pop restored head (INV-5)");
    CHECK(f.live() == 0, "size stays zero, no underflow");
    // repeated failures must not corrupt the counters
    for (int i = 0; i < 5; ++i) (void)f.pop();
    CHECK(f.head == f.tail, "repeated exhaustion leaves head == tail");
  }

  // --- a full allocate / release cycle, and wraparound ---------------------
  {
    Fixture f(4);
    for (int round = 0; round < 5; ++round) {
      std::vector<int32_t> taken;
      for (int i = 0; i < 4; ++i) {
        const PopResult r = f.pop();
        CHECK(r.node != kNoNode, "node available at start of round");
        f.make_resident(r.node);
        taken.push_back(r.node);
      }
      CHECK(f.pop().node == kNoNode, "all four pinned => exhausted");
      for (int32_t n : taken) f.release(n);
      CHECK(f.live() == 4, "all four back in the ring");
    }
    CHECK(f.tail > 4, "counters advanced well past capacity (wraparound)");
  }

  // --- writeback decision comes from the claimed state --------------------
  {
    Fixture f(2);
    const PopResult first = f.pop();
    f.make_resident(first.node);
    f.release(first.node);
    // second pop of the same node: now it holds a row
    (void)f.pop();  // consumes the other EMPTY node
    const PopResult again = f.pop();
    CHECK(again.node == first.node, "the released node came back round");
    CHECK(needs_writeback(again.claimed_from),
          "a node that held a row is written back on eviction");
  }

  // --- stale entry: node re-hit while queued ------------------------------
  {
    Fixture f(4);
    // Cycle node 0 through a step. Releasing puts it back at the *tail*, so
    // three other entries sit ahead of it.
    const PopResult r0 = f.pop();
    CHECK(r0.node == 0, "node 0 first");
    f.make_resident(r0.node);
    f.release(r0.node);
    CHECK(in_ring_of(f.states[0]), "node 0 listed again");

    // Drain those three so the hand arrives at node 0's entry.
    for (int i = 0; i < 3; ++i) {
      CHECK(f.pop().node != 0, "the entries queued ahead of node 0 come first");
    }
    CHECK(f.live() == 1, "only node 0's entry is left");

    // It is hit again just before the hand reaches it, making that entry stale.
    CHECK(try_pin(&f.states[0]), "cache hit on a queued node");

    const PopResult r = f.pop();
    CHECK(r.node == kNoNode, "pinned node is never handed out (INV-1)");
    CHECK(!in_ring_of(f.states[0]), "stale entry dropped and flag cleared");
    CHECK(f.head <= f.tail, "the exhausted pop still restored head");

    // Dropping does not strand it: the cleared flag lets it enqueue again.
    f.release(0);
    CHECK(in_ring_of(f.states[0]), "node re-enqueued after the drop");
    CHECK(f.live() == 1, "and it is back in the ring exactly once");
  }

  // --- a node is never listed twice ---------------------------------------
  {
    Fixture f(4);
    const PopResult r = f.pop();
    f.make_resident(r.node);
    f.release(r.node);
    const int64_t once = f.live();

    // hit + release repeatedly: in_ring is already set, so no new entries
    for (int i = 0; i < 5; ++i) {
      CHECK(try_pin(&f.states[static_cast<size_t>(r.node)]), "re-hit");
      f.release(r.node);
    }
    CHECK(f.live() == once, "repeated hit/release added no ring entries (INV-4)");
  }

  // --- FIFO evicts a hot node; CLOCK spares it ----------------------------
  //
  // Mirrors the A B C D A E trace from the design doc: with capacity 4, node 0
  // is hit just before the hand reaches it.
  {
    auto build = [](Fixture& f) {
      for (int i = 0; i < 4; ++i) {
        const PopResult r = f.pop();
        f.make_resident(r.node);
        f.release(r.node);
      }
      // node 0 is at the head; hit it (sets `used`), then release
      CHECK(try_pin(&f.states[0]), "hot node hit");
      f.release(0);
      CHECK(used_of(f.states[0]), "hit left the recency bit set");
    };

    Fixture fifo(4);
    build(fifo);
    CHECK(fifo.pop(PopPolicy::fifo()).node == 0,
          "FIFO evicts the hot node purely because it is oldest");

    Fixture clock(4);
    build(clock);
    std::vector<int32_t> retry_nodes(4);
    uint32_t retry_count = 0;
    RetryBuffer rb{retry_nodes.data(), &retry_count, 4};
    const PopResult r = clock.pop(PopPolicy::clock_with(1), &rb);
    CHECK(r.node != 0, "CLOCK spares the hot node");
    CHECK(retry_count == 1 && retry_nodes[0] == 0, "hot node parked for requeue");
    CHECK(!used_of(clock.states[0]), "second chance consumed the recency bit");
    CHECK(in_ring_of(clock.states[0]),
          "spared node keeps in_ring: its entry moved, it did not vanish");

    // the follow-up kernel puts it back
    ring_absorb_retry(clock.view(), rb, 0);
    CHECK(clock.live() > 0, "retry buffer absorbed back into the ring");
  }

  // --- second chances are budgeted, so pop always terminates --------------
  //
  // Every entry is hot. An unbudgeted CLOCK would sweep them all into the retry
  // buffer and then report the ring empty while four reclaimable nodes existed.
  {
    Fixture f(4);
    for (int i = 0; i < 4; ++i) {
      const PopResult r = f.pop();
      f.make_resident(r.node);
      f.release(r.node);
      CHECK(try_pin(&f.states[static_cast<size_t>(r.node)]), "hit");
      f.release(r.node);
    }
    for (int i = 0; i < 4; ++i) CHECK(used_of(f.states[static_cast<size_t>(i)]), "all hot");

    std::vector<int32_t> retry_nodes(4);
    uint32_t retry_count = 0;
    RetryBuffer rb{retry_nodes.data(), &retry_count, 4};
    const PopResult r = f.pop(PopPolicy::clock_with(1), &rb);
    CHECK(r.node != kNoNode,
          "budget forces a claim rather than a spurious 'ring empty'");
    CHECK(retry_count == 1, "exactly one second chance was granted");
  }

  // --- retry buffer overflow degrades, never loses a node -----------------
  {
    Fixture f(4);
    for (int i = 0; i < 4; ++i) {
      const PopResult r = f.pop();
      f.make_resident(r.node);
      f.release(r.node);
      CHECK(try_pin(&f.states[static_cast<size_t>(r.node)]), "hit");
      f.release(r.node);
    }
    std::vector<int32_t> retry_nodes(1);
    uint32_t retry_count = 1;  // pretend it is already full
    RetryBuffer rb{retry_nodes.data(), &retry_count, 1};
    const PopResult r = f.pop(PopPolicy::clock_with(8), &rb);
    CHECK(r.node != kNoNode, "full retry buffer still yields a node");
    CHECK(retry_count == 1, "no entry written past the buffer");
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
