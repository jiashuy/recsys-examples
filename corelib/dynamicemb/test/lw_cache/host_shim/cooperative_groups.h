// ---------------------------------------------------------------------------
// A HOST-ONLY STAND-IN FOR <cooperative_groups.h>.  NOT THE REAL THING.
//
// There is one thread here, so every group is a group of one: size() == 1,
// thread_rank() == 0, and shfl() returns what it was given. That is the correct
// degenerate case, and it is also why this shim can say nothing at all about
// the thing warp aggregation exists for — that N lanes collapse into one
// atomic. Under this header they collapse into one because there was only ever
// one.
//
// Kept to the members actually called. coalesced_threads() on real hardware
// captures the currently converged lanes, which is a property no single-threaded
// stand-in can model; anything relying on the size being > 1 has to be checked
// on a device.
// ---------------------------------------------------------------------------
#pragma once

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#error "lw_cache host_shim/cooperative_groups.h was included in a CUDA \
compilation. Remove testing/host_shim from the include path; it exists only \
for the host-only test build."
#endif

namespace cooperative_groups {

class coalesced_group {
 public:
  unsigned int size() const { return 1u; }
  unsigned int thread_rank() const { return 0u; }
  template <typename T>
  T shfl(T value, unsigned int) const {
    return value;
  }
  void sync() const {}
};

inline coalesced_group coalesced_threads() { return coalesced_group{}; }

}  // namespace cooperative_groups
