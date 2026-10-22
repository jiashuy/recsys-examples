// Host-only stand-in for <cuda_fp16.h>.
//
// A float in a struct. It is NOT half precision: nothing here rounds, so a
// value that would lose bits on a device keeps them here. Only enough to let
// src/utils.h compile — no arithmetic operators, deliberately, so that code
// doing real half math fails here instead of quietly computing in float.
#pragma once

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#error "host_shim/cuda_fp16.h was included in a CUDA compilation."
#endif

struct __half {
  float v;
  __half() : v(0.f) {}
  explicit __half(float f) : v(f) {}
};

inline __half __float2half(float f) { return __half(f); }
inline float __half2float(__half h) { return h.v; }

// The real header provides both spellings.
typedef __half half;
