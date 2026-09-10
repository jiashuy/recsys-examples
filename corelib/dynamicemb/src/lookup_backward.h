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
#include <optional>

namespace dyn_emb {

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
  void local_reduce(const at::Tensor &in_grads, at::Tensor &out_grads,
                    const at::Tensor &sorted_key_ids,
                    const at::Tensor &unique_key_ids, cudaStream_t &stream,
                    const PooledLayout &layout = {0, 0, 0, 0, nullptr},
                    bool feature_dims_vec4 = false,
                    const std::optional<at::Tensor> &offsets = std::nullopt,
                    PoolingMode pooling_mode = PoolingMode::kNone,
                    const std::optional<at::Tensor> &weights = std::nullopt);
};

} // namespace dyn_emb
#endif // LOOKUP_BACKWARD_H
