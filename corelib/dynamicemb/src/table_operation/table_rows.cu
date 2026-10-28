/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
******************************************************************************/

#include "../check.h"
#include "table.cuh"
#include "types.cuh"

#include <cuda_runtime.h>
#include <torch/extension.h>

namespace dyn_emb {

// Map table-relative hash slots to the value rows they own.
//
// A slot owns its row for as long as the table lives: the first key to occupy
// a slot takes the next row of its table, and every later occupant -- the key
// that evicts it, or the one that reuses it after an erase -- inherits that
// row. Negative entries are not slots and pass through (see below). Anything
// else must be a slot of the table's main region, and, unless `allocate`, one
// that already owns a row; both are asserted on the device.
__global__ void table_resolve_rows_kernel(
    int32_t *__restrict__ rows, int32_t *__restrict__ row_counters,
    int64_t const *__restrict__ table_bucket_offsets, int64_t bucket_capacity,
    int64_t const *__restrict__ slots, int64_t const *__restrict__ table_ids,
    int64_t *__restrict__ out, int64_t n, bool allocate) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t slot = slots[i];
    if (slot < 0) {
      // Not a slot at all: -1 marks a lookup miss, and -(position + 1) marks an
      // input key the table rejected, reported among the evicted ones. Both
      // encodings are consumed downstream (select_insert_failed_values reads
      // the position back), so they must come out as they went in.
      out[i] = slot;
      continue;
    }
    int64_t tid = table_ids[i];
    int64_t bkt_begin = table_bucket_offsets[tid];
    int64_t main_capacity =
        (table_bucket_offsets[tid + 1] - bkt_begin) * bucket_capacity;
    // Overflow slots sit past the main region and have no row; a table that
    // owns rows has no overflow.
    CUDA_KERNEL_ASSERT(slot < main_capacity);
    int64_t flat = bkt_begin * bucket_capacity + slot;
    int32_t row = rows[flat];
    if (row < 0) {
      // Without `allocate` the slot's key was placed earlier, so it owns a row;
      // finding none means a path put a key here without going through rows.
      CUDA_KERNEL_ASSERT(allocate);
      if (allocate) {
        // Keys in one batch are unique, so no other thread touches this slot.
        row = atomicAdd(row_counters + tid, 1);
        rows[flat] = row;
      }
    }
    out[i] = row;
  }
}

at::Tensor table_resolve_rows(at::Tensor rows, at::Tensor row_counters,
                              at::Tensor table_bucket_offsets,
                              int64_t bucket_capacity, at::Tensor slots,
                              at::Tensor table_ids, bool allocate) {
  TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kInt);
  TORCH_CHECK(row_counters.is_cuda() && row_counters.scalar_type() == at::kInt);
  TORCH_CHECK(slots.is_cuda() && slots.scalar_type() == at::kLong);
  TORCH_CHECK(table_ids.is_cuda() && table_ids.scalar_type() == at::kLong);
  TORCH_CHECK(slots.numel() == table_ids.numel());
  int64_t n = slots.numel();
  at::Tensor out = torch::empty_like(slots);
  if (n == 0)
    return out;
  constexpr int BLOCK = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  table_resolve_rows_kernel<<<(n + BLOCK - 1) / BLOCK, BLOCK, 0, stream>>>(
      rows.data_ptr<int32_t>(), row_counters.data_ptr<int32_t>(),
      table_bucket_offsets.data_ptr<int64_t>(), bucket_capacity,
      slots.data_ptr<int64_t>(), table_ids.data_ptr<int64_t>(),
      out.data_ptr<int64_t>(), n, allocate);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}


// ---------------------------------------------------------------------------
// Rehash support: carrying the rows of reclaimed slots
// ---------------------------------------------------------------------------
//
// An erased key leaves its slot reclaimed, still owning its row. When a table
// is rehashed those rows must survive, but a reclaimed slot has no key to hash
// by, so it is re-placed into any free slot of the new table. Both steps below
// touch only the reclaimed slots (n of them, usually few) beyond one streaming
// pass over the table, so nothing is allocated that scales with the capacity.

constexpr int64_t kEmptyKey = static_cast<int64_t>(0xFFFFFFFFFFFFFFFFull);
constexpr int64_t kReclaimedKey = static_cast<int64_t>(0xFFFFFFFFFFFFFFFEull);

// Collects the rows of the reclaimed slots in the bucket range
// [bkt_begin, bkt_end) that still own a row into `out`, which has room for
// exactly `num_rows` of them: every row below a table's counter sits on a live
// or a reclaimed slot, so the caller knows the count without scanning.
// `keys` is the [num_buckets, bucket_capacity] key column, strided; `rows` is
// flat per slot.
__global__ void collect_reclaimed_rows_kernel(
    int64_t const *__restrict__ keys, int64_t key_stride,
    int32_t const *__restrict__ rows, int64_t bkt_begin, int64_t num_slots,
    int64_t bucket_capacity, int32_t *__restrict__ out, int64_t num_rows,
    int32_t *__restrict__ counter) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_slots; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int64_t bucket = bkt_begin + i / bucket_capacity;
    int64_t iter = i % bucket_capacity;
    if (keys[bucket * key_stride + iter] != kReclaimedKey)
      continue;
    int32_t row = rows[bucket * bucket_capacity + iter];
    if (row < 0)
      continue;
    int32_t pos = atomicAdd(counter, 1);
    CUDA_KERNEL_ASSERT(pos < num_rows);
    out[pos] = row;
  }
}

// Gives each of the n carried rows a free slot of the table region
// [bkt_begin, bkt_end): linear probing over the flattened slots from a start
// spread evenly across the region, claiming an empty slot by CAS and marking it
// reclaimed. Linear probing fills the same set of slots whatever the order the
// threads claim them in, so the key layout does not depend on scheduling.
__global__ void place_reclaimed_rows_kernel(
    int64_t *__restrict__ keys, int64_t key_stride, int32_t *__restrict__ rows,
    int32_t const *__restrict__ carried, int64_t n, int64_t bkt_begin,
    int64_t num_slots, int64_t bucket_capacity) {
  for (int64_t j = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       j < n; j += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // A hole means fewer reclaimed rows were found than the counters promised.
    CUDA_KERNEL_ASSERT(carried[j] >= 0);
    // Row j starts at the head of the j-th of n equal stretches of the region.
    // Spreading the starts keeps the reclaimed slots from piling into the first
    // buckets -- a reclaimed slot is only reused once a bucket has no empty
    // slot left, while its neighbours still do -- and keeps the threads from
    // all CASing the same few slots. The product stays below 2^63 while n is
    // within int32 and the region within 2^32 slots.
    int64_t start = (j * num_slots) / n;
    bool placed = false;
    for (int64_t k = 0; k < num_slots && !placed; ++k) {
      int64_t i = start + k;
      if (i >= num_slots)
        i -= num_slots;
      int64_t bucket = bkt_begin + i / bucket_capacity;
      int64_t iter = i % bucket_capacity;
      int64_t *key = keys + bucket * key_stride + iter;
      if (*reinterpret_cast<volatile int64_t *>(key) != kEmptyKey)
        continue;
      unsigned long long old = atomicCAS(
          reinterpret_cast<unsigned long long *>(key),
          static_cast<unsigned long long>(kEmptyKey),
          static_cast<unsigned long long>(kReclaimedKey));
      if (static_cast<int64_t>(old) == kEmptyKey) {
        rows[bucket * bucket_capacity + iter] = carried[j];
        placed = true;
      }
    }
    // A rehash only grows a table, so there is always room.
    CUDA_KERNEL_ASSERT(placed);
  }
}

static void check_keys_view(at::Tensor const &keys) {
  TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kLong);
  TORCH_CHECK(keys.dim() == 2 && keys.stride(1) == 1);
}

at::Tensor table_collect_reclaimed_rows(at::Tensor keys, at::Tensor rows,
                                        int64_t bkt_begin, int64_t bkt_end,
                                        int64_t bucket_capacity,
                                        int64_t num_rows) {
  check_keys_view(keys);
  TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kInt);
  TORCH_CHECK(num_rows > 0, "nothing to collect: the caller knows the count");
  int64_t num_slots = (bkt_end - bkt_begin) * bucket_capacity;
  // Slots not filled (the caller's count was too high) stay -1.
  at::Tensor out = torch::full({num_rows}, -1, rows.options());
  if (num_slots > 0) {
    at::Tensor counter = torch::zeros({1}, rows.options());
    constexpr int BLOCK = 256;
    int64_t grid = std::min<int64_t>((num_slots + BLOCK - 1) / BLOCK, 1 << 20);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    collect_reclaimed_rows_kernel<<<grid, BLOCK, 0, stream>>>(
        keys.data_ptr<int64_t>(), keys.stride(0), rows.data_ptr<int32_t>(),
        bkt_begin, num_slots, bucket_capacity, out.data_ptr<int32_t>(),
        num_rows, counter.data_ptr<int32_t>());
    DEMB_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return out;
}

void table_place_reclaimed_rows(at::Tensor keys, at::Tensor rows,
                                at::Tensor carried, int64_t bkt_begin,
                                int64_t bkt_end, int64_t bucket_capacity) {
  check_keys_view(keys);
  TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kInt);
  TORCH_CHECK(carried.is_cuda() && carried.scalar_type() == at::kInt);
  int64_t n = carried.numel();
  int64_t num_slots = (bkt_end - bkt_begin) * bucket_capacity;
  if (n == 0)
    return;
  TORCH_CHECK(n <= num_slots, "more reclaimed rows than slots to hold them");
  constexpr int BLOCK = 256;
  int64_t grid = std::min<int64_t>((n + BLOCK - 1) / BLOCK, 1 << 20);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  place_reclaimed_rows_kernel<<<grid, BLOCK, 0, stream>>>(
      keys.data_ptr<int64_t>(), keys.stride(0), rows.data_ptr<int32_t>(),
      carried.data_ptr<int32_t>(), n, bkt_begin, num_slots, bucket_capacity);
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace dyn_emb
