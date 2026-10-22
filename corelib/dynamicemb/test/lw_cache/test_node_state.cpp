// Host-side exercise of the node state word and its transitions.
//
// Single-threaded, so this validates the *protocol* (which transitions are
// legal, what each returns, which bits survive) and not its race-freedom.
#include <cstdio>
#include <initializer_list>

#include "node_state.h"

using namespace lw_cache;

static int failures = 0;
#define CHECK(cond, msg)                                               \
  do {                                                                 \
    if (!(cond)) {                                                     \
      std::printf("  FAIL: %s  (%s:%d)\n", msg, __FILE__, __LINE__);   \
      ++failures;                                                      \
    }                                                                  \
  } while (0)

int main() {
  // --- bit layout is disjoint and covers the word ------------------------
  {
    CHECK((kStateMask | kReservedMask | kInRingMask | kUsedMask | kRefMask) ==
              0xFFFFFFFFu,
          "fields cover the whole word");
    CHECK((kStateMask & kReservedMask) == 0 && (kStateMask & kInRingMask) == 0 &&
              (kStateMask & kUsedMask) == 0 && (kStateMask & kRefMask) == 0 &&
              (kInRingMask & kUsedMask) == 0 && (kInRingMask & kRefMask) == 0 &&
              (kUsedMask & kRefMask) == 0,
          "fields are disjoint");
    CHECK(kRefMax == 134217727u, "ref field is 27 bits");
  }

  // --- accessors round-trip ----------------------------------------------
  {
    for (uint32_t s = 0; s < 4; ++s) {
      for (int ir = 0; ir < 2; ++ir) {
        for (int us = 0; us < 2; ++us) {
          const StateWord w = make_state(NodeState(s), 12345u, ir, us);
          CHECK(uint32_t(state_of(w)) == s, "state round-trips");
          CHECK(ref_of(w) == 12345u, "ref round-trips");
          CHECK(in_ring_of(w) == bool(ir), "in_ring round-trips");
          CHECK(used_of(w) == bool(us), "used round-trips");
        }
      }
    }
    const StateWord init = initial_state();
    CHECK(state_of(init) == kEmpty && ref_of(init) == 0 && in_ring_of(init) &&
              !used_of(init),
          "initial node is EMPTY, ref 0, already in the ring");
  }

  // --- try_pin ------------------------------------------------------------
  {
    StateWord w = make_state(kResident, 0u, /*in_ring=*/true, /*used=*/false);
    CHECK(try_pin(&w), "pin succeeds on RESIDENT");
    CHECK(ref_of(w) == 1u, "pin incremented ref");
    CHECK(used_of(w), "pin set the CLOCK recency bit");
    CHECK(in_ring_of(w), "pin left in_ring alone (entry is now stale)");
    CHECK(state_of(w) == kResident, "pin left state alone");

    CHECK(try_pin(&w) && ref_of(w) == 2u, "pin nests for a second in-flight step");

    for (NodeState s : {kEmpty, kLoading, kEvicting}) {
      StateWord x = make_state(s, 0u, true, false);
      CHECK(!try_pin(&x), "pin fails unless RESIDENT");
      CHECK(ref_of(x) == 0u, "failed pin left ref untouched");
    }

    // A node mid-eviction must not be pinnable: this is the window that would
    // otherwise hand a caller a row about to be reused.
    StateWord evicting = make_state(kEvicting, 0u, false, false);
    CHECK(!try_pin(&evicting), "pin loses to an in-progress claim");
  }

  // --- try_claim ----------------------------------------------------------
  {
    StateWord idle = make_state(kResident, 0u, /*in_ring=*/true, /*used=*/true);
    CHECK(try_claim(&idle, nullptr), "claim succeeds on idle RESIDENT");
    CHECK(state_of(idle) == kEvicting, "claim moved to EVICTING");
    CHECK(in_ring_of(idle) && used_of(idle),
          "claim preserved in_ring/used for the caller to clear");

    StateWord empty = make_state(kEmpty, 0u, true, false);
    CHECK(try_claim(&empty, nullptr), "claim succeeds on EMPTY (first allocation)");

    StateWord pinned = make_state(kResident, 1u, true, false);
    CHECK(!try_claim(&pinned, nullptr), "claim fails while referenced (INV-1)");
    CHECK(state_of(pinned) == kResident, "failed claim left state untouched");

    StateWord loading = make_state(kLoading, 1u, false, false);
    CHECK(!try_claim(&loading, nullptr), "claim fails on LOADING");

    // pin and claim are mutually exclusive on the same word
    StateWord w = make_state(kResident, 0u, true, false);
    CHECK(try_pin(&w), "pin first");
    CHECK(!try_claim(&w, nullptr), "claim then loses");
    StateWord v = make_state(kResident, 0u, true, false);
    CHECK(try_claim(&v, nullptr), "claim first");
    CHECK(!try_pin(&v), "pin then loses");
  }

  // --- writeback decision -------------------------------------------------
  {
    CHECK(!needs_writeback(make_state(kEmpty, 0, true, false)),
          "EMPTY victim carries no data");
    CHECK(needs_writeback(make_state(kResident, 0, true, true)),
          "RESIDENT victim is always written back (INV-9, no dirty bit)");
  }

  // --- loading -> resident ------------------------------------------------
  {
    StateWord w = loading_state();
    CHECK(state_of(w) == kLoading && ref_of(w) == 1u, "fresh node is LOADING, pinned");
    CHECK(!in_ring_of(w), "fresh node is not in the ring");
    CHECK(!used_of(w), "fresh node has not earned a second chance yet");
    mark_resident(&w);
    CHECK(state_of(w) == kResident && ref_of(w) == 1u,
          "mark_resident preserved ref");
  }

  // --- release_ref --------------------------------------------------------
  {
    // ref 2 -> 1: no enqueue
    StateWord w = make_state(kResident, 2u, /*in_ring=*/false, false);
    CHECK(!release_ref(&w), "release from 2 does not enqueue");
    CHECK(ref_of(w) == 1u, "ref decremented");
    CHECK(!in_ring_of(w), "in_ring untouched while still referenced");

    // ref 1 -> 0, not yet listed: this thread enqueues
    CHECK(release_ref(&w), "1->0 with in_ring clear enqueues");
    CHECK(ref_of(w) == 0u, "ref reached zero");
    CHECK(in_ring_of(w), "in_ring now set");

    // already listed: re-pinning and releasing must NOT enqueue a second time,
    // or the ring would hold two entries for one node
    CHECK(try_pin(&w), "re-hit a node sitting in the ring");
    CHECK(!release_ref(&w), "1->0 with in_ring already set does not enqueue");
    CHECK(in_ring_of(w), "in_ring stays set");

    // after the stale entry is consumed and discarded, the node may enqueue again
    clear_in_ring(&w);
    CHECK(!in_ring_of(w), "clear_in_ring cleared the flag");
    CHECK(try_pin(&w), "pin again");
    CHECK(release_ref(&w), "enqueues once the flag was cleared (no leak)");
  }

  // --- CLOCK recency bit --------------------------------------------------
  {
    StateWord w = make_state(kResident, 0u, true, /*used=*/false);
    CHECK(!take_used(&w), "cold node has no second chance");
    CHECK(try_pin(&w), "a hit sets used");
    CHECK(take_used(&w), "hit node earns a second chance");
    CHECK(!used_of(w), "take_used cleared the bit");
    CHECK(!take_used(&w), "second sweep finds it cold");
    CHECK(ref_of(w) == 1u && state_of(w) == kResident,
          "take_used touched nothing else");
  }

  if (failures == 0) {
    std::printf("all host-side checks passed\n");
    return 0;
  }
  std::printf("%d check(s) failed\n", failures);
  return 1;
}
