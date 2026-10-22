/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
******************************************************************************/

// ---------------------------------------------------------------------------
// Where device memory comes from.
//
// Every component that owns device memory takes an allocator, so that a cache
// embedded in a PyTorch job can draw from torch's caching allocator rather than
// calling cudaMalloc behind its back. That matters for three reasons, none of
// them about speed:
//
//   * memory the cache holds shows up in torch.cuda.memory_allocated() and in
//     memory snapshots, instead of being a gap between what torch reports and
//     what nvidia-smi does
//   * an out-of-memory error names the cache, rather than surfacing somewhere
//     unrelated that happened to allocate next
//   * freed blocks return to torch's pool and get reused, instead of each
//     cudaFree handing memory back to the driver for torch to re-acquire
//
// ---------------------------------------------------------------------------
// The contract
// ---------------------------------------------------------------------------
//
// An allocator provides:
//
//   using Buffer = ...;                  // owns a device allocation
//   Buffer allocate(size_t bytes) const; // zero bytes is legal, yields empty
//
// A buffer provides:
//
//   void*  data()  const;   // device pointer, nullptr when empty
//   size_t bytes() const;
//
// and releases its allocation when destroyed. It must be movable; whether it is
// copyable is up to the implementation (the torch one is, because at::Tensor is
// reference counted).
//
// Allocators may hold state — a device index, a stream, a pool — so they are
// stored rather than instantiated at each use. They only ever live on the host,
// so their size costs nothing.
//
// Alignment: buffers are aligned to at least 8 bytes, which is what the node
// arrays need. Both implementations here give considerably more.
// ---------------------------------------------------------------------------

#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

// Detected rather than configured, so the host tests — which build with a
// stand-in for the CUDA runtime and no torch at all — need no special flags.
//
// This must sit outside the namespace: an #include inside one would drop every
// ATen declaration into embedding_cache::.
#if defined(__has_include)
#if __has_include(<ATen/ATen.h>)
#define EMBEDDING_CACHE_HAS_TORCH 1
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#endif
#endif

namespace embedding_cache {

// ---------------------------------------------------------------------------
// CudaAllocator — cudaMalloc / cudaFree
// ---------------------------------------------------------------------------
//
// The default, and what the host tests use. Fine standalone; inside a PyTorch
// process prefer TorchAllocator, so the two do not compete for memory without
// either knowing about the other.

class CudaBuffer {
 public:
  CudaBuffer() = default;

  explicit CudaBuffer(size_t bytes) : bytes_(bytes) {
    if (bytes == 0) return;
    const cudaError_t e = cudaMalloc(&data_, bytes);
    if (e != cudaSuccess) {
      throw std::runtime_error("embedding_cache: cudaMalloc of " +
                               std::to_string(bytes) +
                               " bytes failed: " + cudaGetErrorString(e));
    }
  }

  ~CudaBuffer() {
    if (data_) cudaFree(data_);
  }

  CudaBuffer(CudaBuffer&& o) noexcept
      : data_(o.data_), bytes_(o.bytes_) {
    o.data_ = nullptr;
    o.bytes_ = 0;
  }
  CudaBuffer& operator=(CudaBuffer&& o) noexcept {
    if (this != &o) {
      if (data_) cudaFree(data_);
      data_ = o.data_;
      bytes_ = o.bytes_;
      o.data_ = nullptr;
      o.bytes_ = 0;
    }
    return *this;
  }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  void* data() const { return data_; }
  size_t bytes() const { return bytes_; }

 private:
  void* data_ = nullptr;
  size_t bytes_ = 0;
};

struct CudaAllocator {
  using Buffer = CudaBuffer;
  Buffer allocate(size_t bytes) const { return Buffer(bytes); }
};

// ---------------------------------------------------------------------------
// TorchAllocator — at::Tensor
// ---------------------------------------------------------------------------
//
// Compiled only where ATen is on the include path; see the detection above.
//
// ---------------------------------------------------------------------------
// Why this one takes a stream and CudaAllocator does not
// ---------------------------------------------------------------------------
//
// torch's caching allocator is stream-aware in a way cudaMalloc is not. Every
// block it hands out is tagged with the stream that was current when it was
// allocated, and on free it may hand that block straight back to the pool for
// reuse *on that stream* — with no synchronisation, because work on one stream
// is ordered. Memory allocated on stream A and still being read by a kernel on
// stream B can therefore be reissued to somebody else while B is mid-read.
//
// The rest of this repository never meets that problem: it allocates and uses
// on at::cuda::getCurrentCUDAStream() throughout, so the tag is always right.
// This cache is the first thing here that deliberately runs on a stream of its
// own — the prefetch stream — which makes the mismatch reachable. So the
// allocator takes that stream and makes it current for the at::empty, and the
// tag is right from the start.
//
// record_stream() below is the other half of torch's answer to the same
// problem. It covers a block being freed while another stream still has work
// queued against it: it tells the allocator that a second stream touches this
// block, so freeing it records an event on that stream instead of returning the
// memory to the pool straight away.
//
// The structural allocations do not need it — bucket heads, the node array, the
// row pool are taken once when the cache is built and released once when it is
// destroyed, so nothing is freed mid-flight. What needs it is per-step scratch:
// a miss list, a compaction buffer, anything a component allocates and releases
// inside a step while other streams are still running. This allocator is not
// the hash map's; every component that owns device memory takes one.
//
// It is on TorchBuffer and not in the Buffer contract at the top of this file.
// Putting it there would oblige every allocator to carry a no-op, and nothing
// generic calls it yet — a component that allocates scratch through a template
// parameter will be the thing that makes it worth widening.

#ifdef EMBEDDING_CACHE_HAS_TORCH

class TorchBuffer {
 public:
  TorchBuffer() = default;

  TorchBuffer(size_t bytes, int device_index, cudaStream_t stream) {
    if (bytes == 0) return;
    // Allocated as int64 rather than uint8 so the pointer is 8-byte aligned by
    // construction, which is what the node arrays need. Relying on the caching
    // allocator's internal alignment would work today and is not a promise.
    const int64_t elems = static_cast<int64_t>((bytes + 7) / 8);
    const auto opts =
        at::TensorOptions().dtype(at::kLong).device(at::kCUDA, device_index);
    if (stream != nullptr) {
      // Makes the block's stream tag the cache's stream rather than whatever
      // happened to be current on the thread that built the cache.
      const c10::cuda::CUDAStreamGuard guard(c10::cuda::getStreamFromExternal(
          stream, static_cast<c10::DeviceIndex>(device_index)));
      t_ = at::empty({elems}, opts);
    } else {
      t_ = at::empty({elems}, opts);
    }
    bytes_ = bytes;
  }

  void* data() const { return t_.defined() ? t_.data_ptr() : nullptr; }
  size_t bytes() const { return bytes_; }

  // Declare that `stream` also reads or writes this block.
  //
  // Needed once for every stream that touches the memory beyond the one it was
  // allocated on, and only worth calling for a buffer that is actually freed
  // while work may still be outstanding — per-step scratch, not the structural
  // arrays. Without it the allocator is free to reissue the block on free while
  // that stream still has kernels queued against it.
  //
  // Sticky: once recorded, every later free of this block waits on that stream.
  void record_stream(cudaStream_t stream) const {
    if (!t_.defined() || stream == nullptr) return;
    t_.record_stream(c10::cuda::getStreamFromExternal(
        stream, static_cast<c10::DeviceIndex>(t_.get_device())));
  }

  // The underlying tensor, for callers that want to hand the memory to torch —
  // exposing a row pool as a tensor, say, without copying.
  const at::Tensor& tensor() const { return t_; }

 private:
  at::Tensor t_;   // reference counted, so this type is copyable as well
  size_t bytes_ = 0;
};

class TorchAllocator {
 public:
  using Buffer = TorchBuffer;

  // -1 means the current device at the time of allocation. A null stream means
  // "whatever is current", which is right for a cache that lives on the default
  // stream and wrong for one that does not — see the note above.
  explicit TorchAllocator(int device_index = -1, cudaStream_t stream = nullptr)
      : device_(device_index), stream_(stream) {}

  Buffer allocate(size_t bytes) const {
    int dev = device_;
    if (dev < 0) {
      cudaGetDevice(&dev);
    }
    return Buffer(bytes, dev, stream_);
  }

  int device_index() const { return device_; }
  cudaStream_t stream() const { return stream_; }

 private:
  int device_ = -1;
  cudaStream_t stream_ = nullptr;
};

#endif  // EMBEDDING_CACHE_HAS_TORCH

}  // namespace embedding_cache
