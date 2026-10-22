// Host-only stand-in for the CUDA driver API. See cuda_runtime.h for the
// caveat that applies to all of this directory.
//
// src/check.h is the only thing that reaches for the driver API, and it wants
// exactly two names. Nothing else belongs here.
#pragma once

#if defined(__CUDACC__) || defined(__CUDA_ARCH__)
#error "host_shim/cuda.h was included in a CUDA compilation."
#endif

typedef int CUresult;
enum { CUDA_SUCCESS = 0 };

inline CUresult cuGetErrorString(CUresult, const char** str) {
  *str = "stub driver error";
  return CUDA_SUCCESS;
}
