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

#include "../check.h"
#include "table_kernels.cuh"
#include "torch_utils.h"
#include <cuda/std/tuple>
#include <iostream>
#include <torch/extension.h>
#include <type_traits>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

namespace dyn_emb {

void table_lookup(at::Tensor table_storage, int bucket_capacity,
                  at::Tensor input_keys, std::optional<at::Tensor> input_scores,
                  at::Tensor output_values, at::Tensor founds,
                  bool allow_score_update, bool allow_score_accum,
                  bool return_score, bool return_value,
                  EvictPolicy evict_policy) {
  if (evict_policy == EvictPolicy::kLfu and allow_score_accum == false) {
    evict_policy = EvictPolicy::kCustomized;
  }
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  int64_t num_total = input_keys.size(0);
  auto key_type = scalartype_to_datatype(input_keys.dtype().toScalarType());
  auto val_type = scalartype_to_datatype(output_values.dtype().toScalarType());

  auto input_keys_ = input_keys.data_ptr();
  auto input_scores_ =
      input_scores.has_value()
          ? reinterpret_cast<ScoreType *>(input_scores.value().data_ptr())
          : nullptr;
  auto output_values_ = return_value ? output_values.data_ptr() : nullptr;
  auto founds_ = founds.data_ptr<bool>();

  constexpr int BLOCK_SIZE = 256;
  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type, KeyType, [&] {
    DISPATCH_OFFSET_INT_TYPE(val_type, ValType, [&] {
      using BucketLayout = TypeInfo<KeyType, ValType, ScoreType, DigestType>;
      int bucket_bytes = bucket_capacity * BucketLayout::total_size;
      int64_t num_buckets =
          table_storage.numel() * table_storage.element_size() / bucket_bytes;

      using Bucket = SlotsBucket<KeyType, ValType, ScoreType, DigestType,
                                 SlotsBucketTraits<DigestType>>;

      using Table = BucketHashingTable<Bucket>;

      auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                         num_buckets, bucket_capacity);

      using Argument = ScoreArgument<ScoreType>;
      Argument score_arg;
      score_arg.scores = input_scores_;

      auto lookup_kernel = [&] {
        if (allow_score_update) {
          switch (evict_policy) {
          case EvictPolicy::kLru: {
            if (return_score) {
              return table_lookup_kernel<Table, LruScorePolicy<ScoreType, true>,
                                         KernelTraits<1>>;
            } else {
              return table_lookup_kernel<
                  Table, LruScorePolicy<ScoreType, false>, KernelTraits<1>>;
            }
          }
          case EvictPolicy::kLfu: {
            if (return_score) {
              return table_lookup_kernel<Table, LfuScorePolicy<ScoreType, true>,
                                         KernelTraits<1>>;
            } else {
              return table_lookup_kernel<
                  Table, LfuScorePolicy<ScoreType, false>, KernelTraits<1>>;
            }
          }
          case EvictPolicy::kCustomized: {
            if (return_score) {
              return table_lookup_kernel<Table,
                                         CustomizedScorePolicy<ScoreType, true>,
                                         KernelTraits<1>>;
            } else {
              return table_lookup_kernel<
                  Table, CustomizedScorePolicy<ScoreType, false>,
                  KernelTraits<1>>;
            }
          }
          default: {
            throw std::runtime_error("Unsupported evict policy");
          }
          }
        } else {
          if (return_score) {
            return table_lookup_kernel<Table, ReadScorePolicy<ScoreType, true>,
                                       KernelTraits<1>>;
          } else {
            return table_lookup_kernel<Table, ReadScorePolicy<ScoreType, false>,
                                       KernelTraits<1>>;
          }
        }
      }();
      lookup_kernel<<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                      stream>>>(
          table, num_total, reinterpret_cast<KeyType *>(input_keys_),
          output_values_ != nullptr
              ? reinterpret_cast<ValType *>(output_values_)
              : nullptr,
          score_arg, founds_);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void table_insert(at::Tensor table_storage, int bucket_capacity,
                  at::Tensor bucket_sizes, at::Tensor input_keys,
                  at::Tensor input_scores, at::Tensor values,
                  bool allow_value_reuse, bool allow_score_accum) {}

void table_insert_and_evict(at::Tensor table_storage, int bucket_capacity,
                            at::Tensor bucket_sizes, at::Tensor input_keys,
                            at::Tensor input_scores, at::Tensor values,
                            bool allow_value_reuse, bool allow_score_accum,
                            at::Tensor num_evicted, at::Tensor evicted_keys,
                            std::optional<at::Tensor> evicted_values,
                            at::Tensor evicted_scores) {}

at::Tensor create_sub_tensor(const at::Tensor &original_tensor,
                             int64_t offset) {
  if (offset < 0 || offset >= original_tensor.numel()) {
    throw std::runtime_error("Invalid offset");
  }

  void *data_ptr =
      original_tensor.data_ptr() + offset * original_tensor.element_size();

  int64_t new_size = original_tensor.numel() - offset;

  at::Tensor new_tensor =
      at::from_blob(data_ptr, {new_size}, original_tensor.options());

  return new_tensor;
}

inline int get_dtype_size_by_scalar_type(torch::ScalarType scalar_type) {
  switch (scalar_type) {
  case torch::kUInt8:
    return 1;
  case torch::kInt8:
    return 1;
  case torch::kInt16:
    return 2;
  case torch::kInt32:
    return 4;
  case torch::kInt64:
    return 8;
  case torch::kFloat32:
    return 4;
  case torch::kFloat64:
    return 8;
  case torch::kBool:
    return 1;
  case torch::kBFloat16:
    return 2;
  case torch::kFloat16:
    return 2;
  case torch::kUInt16:
    return 2;
  case torch::kUInt32:
    return 4;
  case torch::kUInt64:
    return 8;
  default:
    return -1;
  }
}

std::vector<at::Tensor> table_partition(at::Tensor storage, int bucket_capacity,
                                        int num_buckets,
                                        std::vector<torch::Dtype> dtypes) {
  if (dtypes.size() != 4) {
    throw std::runtime_error(
        "Provides four types of key, value, score, digest.");
  }
  auto score_type = static_cast<torch::ScalarType>(dtypes[2]);
  auto digest_type = static_cast<torch::ScalarType>(dtypes[3]);
  if (score_type != torch::kUInt64) {
    throw std::runtime_error("Score type has to be torch.uint64.");
  }
  if (digest_type != torch::kUInt8) {
    throw std::runtime_error("Score type has to be torch.uint8.");
  }

  auto key_type = static_cast<torch::ScalarType>(dtypes[0]);
  auto val_type = static_cast<torch::ScalarType>(dtypes[1]);

  auto key_type_ = scalartype_to_datatype(key_type);
  auto val_type_ = scalartype_to_datatype(val_type);

  std::vector<at::Tensor> result;
  result.reserve(4);

  DISPATCH_INTEGER_DATATYPE_FUNCTION(key_type_, KeyType, [&] {
    DISPATCH_OFFSET_INT_TYPE(val_type_, ValType, [&] {
      using BucketLayout = TypeInfo<KeyType, ValType, ScoreType, DigestType>;
      int bucket_bytes = bucket_capacity * BucketLayout::total_size;
      if (bucket_bytes * num_buckets !=
          storage.numel() * storage.element_size()) {
        throw std::runtime_error(
            "Storage size mismatched with bucket_bytes * num_buckets");
      }

      // keys
      int stride = bucket_bytes / get_dtype_size_by_scalar_type(key_type);
      void *raw_data =
          storage.data_ptr() + bucket_capacity * BucketLayout::get_offset<0>();
      result.push_back(at::from_blob(raw_data, {num_buckets, bucket_capacity},
                                     {stride, 1},
                                     storage.options().dtype(dtypes[0])));

      // values
      stride = bucket_bytes / get_dtype_size_by_scalar_type(val_type);
      raw_data =
          storage.data_ptr() + bucket_capacity * BucketLayout::get_offset<1>();
      result.push_back(at::from_blob(raw_data, {num_buckets, bucket_capacity},
                                     {stride, 1},
                                     storage.options().dtype(dtypes[1])));

      // scores
      stride = bucket_bytes / get_dtype_size_by_scalar_type(score_type);
      raw_data =
          storage.data_ptr() + bucket_capacity * BucketLayout::get_offset<2>();
      result.push_back(at::from_blob(raw_data, {num_buckets, bucket_capacity},
                                     {stride, 1},
                                     storage.options().dtype(dtypes[2])));

      // digests
      stride = bucket_bytes / get_dtype_size_by_scalar_type(digest_type);
      raw_data =
          storage.data_ptr() + bucket_capacity * BucketLayout::get_offset<3>();
      result.push_back(at::from_blob(raw_data, {num_buckets, bucket_capacity},
                                     {stride, 1},
                                     storage.options().dtype(dtypes[3])));
    });
  });

  return result;
}

std::vector<at::Tensor> tensor_partition(at::Tensor input,
                                         std::vector<int64_t> byte_range,
                                         std::vector<torch::Dtype> dtypes) {
  int num_partition = byte_range.size() - 1;
  std::vector<at::Tensor> result;
  result.reserve(num_partition);
  for (int i = 0; i < num_partition; i++) {
    auto raw_data = input.data_ptr() + byte_range[i];
    int64_t partition_size = byte_range[i + 1] - byte_range[i];
    auto scalar_type = static_cast<torch::ScalarType>(dtypes[i]);
    partition_size =
        partition_size / get_dtype_size_by_scalar_type(scalar_type);
    result.push_back(at::from_blob(raw_data, {partition_size},
                                   input.options().dtype(dtypes[i])));
  }
  return result;
}

} // namespace dyn_emb

namespace py = pybind11;

void bind_table_operation(py::module &m) {
  m.def("tensor_partition", &dyn_emb::tensor_partition,
        "split the tensor into several sub-partitions.", py::arg("input"),
        py::arg("byte_range"), py::arg("dtypes"));
  m.def("table_partition", &dyn_emb::table_partition,
        "split the tensor into several sub-partitions.", py::arg("storage"),
        py::arg("bucket_capacity"), py::arg("num_buckets"), py::arg("dtypes"));
  m.def("table_lookup", &dyn_emb::table_lookup, "lookup the table",
        py::arg("table_storage"), py::arg("bucket_capacity"),
        py::arg("input_keys"), py::arg("input_scores"),
        py::arg("output_values"), py::arg("founds"),
        py::arg("allow_score_update"), py::arg("allow_score_accum"),
        py::arg("return_score"), py::arg("return_value"),
        py::arg("evict_policy"));
  m.def("table_insert", &dyn_emb::table_insert, "insert into the table",
        py::arg("table_storage"), py::arg("bucket_capacity"),
        py::arg("bucket_sizes"), py::arg("input_keys"), py::arg("input_scores"),
        py::arg("values"), py::arg("allow_value_reuse"),
        py::arg("allow_score_accum"));
  m.def("table_insert_and_evict", &dyn_emb::table_insert_and_evict,
        "insert into the table", py::arg("table_storage"),
        py::arg("bucket_capacity"), py::arg("bucket_sizes"),
        py::arg("input_keys"), py::arg("input_scores"), py::arg("values"),
        py::arg("allow_value_reuse"), py::arg("allow_score_accum"),
        py::arg("num_evicted"), py::arg("evicted_keys"),
        py::arg("evicted_values"), py::arg("evicted_scores"));
}
