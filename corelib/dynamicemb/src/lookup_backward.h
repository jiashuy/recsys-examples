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
// row, and getting them there means sorting.  Carrying the weight as part of
// the sort's value is free: the value used to be a bare gather id typed like
// the inverse indices, which segmented_unique emits as int64, so it was 8
// bytes already.  Sorting a separate float array by the same keys, in
// contrast, is an entire second radix sort.  Unweighted lookups store 1.0f,
// and multiplying by it is exact in fp32, so they take the identical path at
// the identical cost and need no separate kernel instantiation.
//
// grad_index is int32: reduce_grads checks up front that both the key count
// and the bag count fit in it.
//
// POD by design, and 8-byte aligned so cub's radix sort handles it exactly
// like a uint64_t value.
struct alignas(8) GradInfo {
  int32_t grad_index;
  float weight;
};
static_assert(sizeof(GradInfo) == 8, "GradInfo must stay 8 bytes");
static_assert(alignof(GradInfo) == 8, "GradInfo must stay 8-byte aligned");

class LocalReduce {
private:
  c10::Device device_;
  int64_t num_key_;
  int64_t len_vec_;
  DataType key_type_;
  DataType id_type_;
  DataType accum_type_;

  at::Tensor partial_buffer;
  at::Tensor partial_unique_ids;

  static constexpr int32_t WarpSize = 32;

public:
  LocalReduce(c10::Device &device, int64_t num_key, int64_t len_vec,
              DataType id_type, DataType accum_type);

  // Unified reduce.  Under mixed dims the source is grads[B, total_D] and a
  // feature's columns come from layout.col_begin/col_width; otherwise the
  // source is addressed uniformly.  MEAN scaling is fused either way.
  // len_vec_ must be set to max_D when the layout has mixed dims.  A sequence
  // reduce is the degenerate case -- every key its own bag -- so it passes no
  // offsets and the layout goes unread.
  //
  // sorted_grad_infos holds one GradInfo per key, already sorted by
  // unique_key_ids; pooling weights ride in on it, so there is no separate
  // weights argument.  It is an int32 tensor of 2*num_key elements reading as
  // GradInfo[num_key] -- an int32 pair rather than an opaque byte blob so it
  // stays inspectable from the Python side.
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
