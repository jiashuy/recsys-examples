/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

#pragma once
#include "torch_utils.h"
#include "utils.h"
#include <cstdint>
#include <cub/cub.cuh>

namespace dyn_emb {

struct SegmentedSortDevice {
public:
  using seg_id_t = int32_t;
  SegmentedSortDevice() = default;

  SegmentedSortDevice(c10::Device &device, int64_t key_num, int32_t segment_num,
                      DataType key_type, DataType value_type);

  void operator()(
      const at::Tensor &keys_in, at::Tensor &values_in,
      const at::Tensor &segment_offsets, at::Tensor &keys_out,
      at::Tensor &values_out,
      at::Tensor
          &segment_ids_out, // indicate keys_out[i] belongs to which segment.
      cudaStream_t &stream,
      bool set_input_value_to_idx = false, // use when indirect sorting.
      bool set_output_segment_ids =
          true); // whether set segment_ids_out or not.

private:
  c10::Device device_;
  int64_t key_num_;
  int32_t segment_num_;
  DataType key_type_;
  DataType value_type_;

  uint64_t cub_sort_temp_bytes_ = 0;
  at::Tensor cub_sort_temp_buffer_; // void
  // The following two members are used when cub version >= 200200.
  // CUB_VERSION is defined in cuh so do not use it here.
  at::Tensor segmented_keys_in_;  // composed
  at::Tensor segmented_keys_out_; // composed
  bool need_compose_flag_;
  bool set_out_segment_ids_;
};

struct SegmentedUniqueDevice {
  using seg_id_t = int32_t;
  c10::Device device_;
  int64_t num_key_;
  DataType key_type_;
  DataType id_type_;
  at::Tensor key_flag_buffer_; // uint32_t

  uint64_t cub_scan_temp_bytes_ = 0;
  at::Tensor cub_scan_temp_buffer_; // void

  SegmentedUniqueDevice() = default;

  SegmentedUniqueDevice(c10::Device &device, int64_t num_key, DataType key_type,
                        DataType id_type);

  void operator()(
      const at::Tensor &sorted_keys, const at::Tensor &sorted_segment_ids,
      at::Tensor &unique_keys,
      at::Tensor &unique_key_ids, // mapping from sorted keys to unique keys
      cudaStream_t &stream);
};

} // namespace dyn_emb