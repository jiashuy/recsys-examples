// Host-only stand-in for <cuda_bf16.h>. Same caveat as cuda_fp16.h: a float in
// a struct, no rounding, no arithmetic.
#pragma once

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#error "host_shim/cuda_bf16.h was included in a CUDA compilation."
#endif

struct __nv_bfloat16 {
  float v;
  __nv_bfloat16() : v(0.f) {}
  explicit __nv_bfloat16(float f) : v(f) {}
};

inline __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16(f); }
inline float __bfloat162float(__nv_bfloat16 b) { return b.v; }

// The real header provides both spellings; src/utils.h uses the short one.
typedef __nv_bfloat16 nv_bfloat16;
