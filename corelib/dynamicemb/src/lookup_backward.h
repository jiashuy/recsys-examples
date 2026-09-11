/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#ifndef LOOKUP_BACKWARD_H
#define LOOKUP_BACKWARD_H
#include "index_calculation.h"
#include "pooled_layout.cuh"
#include "utils.h"
#include <cstdint>
#include <optional>

namespace dyn_emb {

// One key's contribution to the wgrad reduce: the row of ``grads`` it reads,
// and the scale to apply to that row.
//
// The two travel together because the reduce needs the keys grouped by unique
// row, and getting them there means sorting.  Carrying the weight as the
// sort's value rather than sorting a separate float array by the same keys
// saves an entire second radix sort.  Unweighted lookups store 1.0f, and
// multiplying by it is exact in fp32, so they take the identical path and need
// no separate kernel instantiation.
//
// grad_index is IndexType -- the same type as the inverse indices and the
// offsets -- so no row can be truncated on its way through the sort.
//
// Packed to 4, not padded: IndexType is 64-bit throughout this codebase
// (DISPATCH_INTEGER_DATATYPE_FUNCTION covers only Int64/UInt64), so the
// natural layout would waste 4 bytes per key on tail padding.  12 bytes at
// alignment 4 means cub moves three 32-bit words per item instead of two
// 64-bit ones; pack(1) is deliberately avoided, as it would degrade the
// shared-memory exchange to byte granularity.
#pragma pack(push, 4)
template <typename IndexType> struct GradInfo {
  IndexType grad_index;
  float weight;
};
#pragma pack(pop)
static_assert(sizeof(GradInfo<int64_t>) == 12, "GradInfo must stay packed");
static_assert(alignof(GradInfo<int64_t>) == 4, "GradInfo must stay 4-aligned");

// IndexType -> the DataType its tensors are allocated with.  Only the two
// types DISPATCH_INTEGER_DATATYPE_FUNCTION can produce are instantiated.
template <typename T> struct IndexDataType;
template <> struct IndexDataType<int64_t> {
  static constexpr DataType value = DataType::Int64;
};
template <> struct IndexDataType<uint64_t> {
  static constexpr DataType value = DataType::UInt64;
};

// Templated on the index type so the offsets, the inverse indices and
// GradInfo::grad_index are the same type by construction rather than by
// convention.  reduce_grads converts the offsets when they disagree.
// Explicitly instantiated in lookup_backward.cu for both index types, which
// keeps the reduce kernels out of every translation unit that includes this.
template <typename IndexType> class LocalReduce {
private:
  c10::Device device_;
  int64_t num_key_;
  int64_t len_vec_;
  DataType accum_type_;

  at::Tensor partial_buffer;
  at::Tensor partial_unique_ids;

  static constexpr int32_t WarpSize = 32;

public:
  LocalReduce(c10::Device &device, int64_t num_key, int64_t len_vec,
              DataType accum_type);

  // Unified reduce.  Under mixed dims the source is grads[B, total_D] and a
  // feature's columns come from layout.col_begin/col_width; otherwise the
  // source is addressed uniformly.  MEAN scaling is fused either way.
  // len_vec_ must be set to max_D when the layout has mixed dims.  A sequence
  // reduce is the degenerate case -- every key its own bag -- so it passes no
  // offsets and the layout goes unread.
  //
  // sorted_grad_infos holds one GradInfo<IndexType> per key, already sorted by
  // unique_key_ids; pooling weights ride in on it, so there is no separate
  // weights argument.  Packing makes it a byte tensor of
  // num_key * sizeof(GradInfo<IndexType>) elements; reduce_grads owns its
  // validity, so nothing is re-checked here.
  void local_reduce(const at::Tensor &in_grads, at::Tensor &out_grads,
                    const at::Tensor &sorted_grad_infos,
                    const at::Tensor &unique_key_ids, cudaStream_t &stream,
                    const PooledLayout &layout = {0, 0, 0, 0, nullptr},
                    bool feature_dims_vec4 = false,
                    const std::optional<at::Tensor> &offsets = std::nullopt,
                    PoolingMode pooling_mode = PoolingMode::kNone);
};

} // namespace dyn_emb
#endif // LOOKUP_BACKWARD_H
