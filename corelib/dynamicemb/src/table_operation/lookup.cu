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

#include "kernels.cuh"
#include "table.cuh"

namespace dyn_emb {

void table_lookup_single_score(at::Tensor table_storage,
                               std::vector<torch::Dtype> dtypes,
                               int64_t bucket_capacity, at::Tensor keys,
                               std::vector<std::optional<at::Tensor>> scores,
                               std::vector<ScorePolicyType> policy_types,
                               std::vector<bool> is_returns, at::Tensor founds,
                               std::optional<at::Tensor> values) {
  using DigestType = uint8_t;
  using ValueType = int64_t;

  auto key_type = get_data_type(keys);
  auto score_type = get_data_type(dtypes[2]);

  auto founds_ = founds.data_ptr<bool>();

  uint64_t score_code = get_scores_code(policy_types, is_returns);

  auto stream = at::cuda::getCurrentCUDAStream().stream();

  int64_t num_total = keys.size(0);

  constexpr int BLOCK_SIZE = 256;

  DISPATCH_KEY_TYPE(key_type, KeyType, [&] {
    DISPATCH_SCORE_TYPE(score_type, ScoreScalar, [&] {
      auto keys_ = get_pointer<KeyType>(keys);
      auto scores_ = get_pointer<ScoreScalar>(scores[0]);
      auto values_ = get_pointer<ValueType>(values);

      using Policy = ScorePolicy<ScoreScalar>;
      using Arg = typename Policy::ScoreArgument;
      auto score_arg = Arg(scores_, score_code);

      using Score = ScoreTuple<ScoreScalar>;
      constexpr int64_t total_size =
          sizeof(KeyType) + sizeof(DigestType) + Score::size();
      int64_t bucket_bytes = bucket_capacity * total_size;
      int64_t num_buckets =
          table_storage.numel() * table_storage.element_size() / bucket_bytes;

      /// TODO: score type, kernel traits
      using Bucket = LinearBucket<KeyType, DigestType, SingleScore,
                                  LinearBucketTraits<DigestType>>;

      using Table = LinearBucketTable<Bucket, ValueType>;

      auto table = Table(reinterpret_cast<uint8_t *>(table_storage.data_ptr()),
                         num_buckets, bucket_capacity);

      Argument score_arg;
      score_arg.scores = input_scores_;

      auto lookup_kernel = [&] {
        if (allow_score_update) {
          switch (evict_policy) {
          case EvictPolicy::kLru: {
            if (return_score) {
              return table_lookup_kernel<
                  Table, LruScorePolicy<ScoreScalar, true>, KernelTraits<1>>;
            } else {
              return table_lookup_kernel<
                  Table, LruScorePolicy<ScoreScalar, false>, KernelTraits<1>>;
            }
          }
          case EvictPolicy::kLfu: {
            if (return_score) {
              return table_lookup_kernel<
                  Table, LfuScorePolicy<ScoreScalar, true>, KernelTraits<1>>;
            } else {
              return table_lookup_kernel<
                  Table, LfuScorePolicy<ScoreScalar, false>, KernelTraits<1>>;
            }
          }
          case EvictPolicy::kCustomized: {
            if (return_score) {
              return table_lookup_kernel<
                  Table, CustomizedScorePolicy<ScoreScalar, true>,
                  KernelTraits<1>>;
            } else {
              return table_lookup_kernel<
                  Table, CustomizedScorePolicy<ScoreScalar, false>,
                  KernelTraits<1>>;
            }
          }
          default: {
            throw std::runtime_error("Unsupported evict policy");
          }
          }
        } else {
          if (return_score) {
            return table_lookup_kernel<
                Table, ReadScorePolicy<ScoreScalar, true>, KernelTraits<1>>;
          } else {
            return table_lookup_kernel<
                Table, ReadScorePolicy<ScoreScalar, false>, KernelTraits<1>>;
          }
        }
      }();
      lookup_kernel<<<(num_total + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0,
                      stream>>>(
          table, num_total, reinterpret_cast<KeyType *>(keys_),
          output_values_ != nullptr
              ? reinterpret_cast<ValType *>(output_values_)
              : nullptr,
          score_arg, founds_);
    });
  });
  DEMB_CUDA_KERNEL_LAUNCH_CHECK();
}

void table_lookup(at::Tensor table_storage, std::vector<torch::Dtype> dtypes,
                  int64_t bucket_capacity, at::Tensor keys,
                  std::vector<std::optional<at::Tensor>> scores,
                  std::vector<ScorePolicy> policies,
                  std::vector<bool> is_returns, at::Tensor founds,
                  std::optional<at::Tensor> values) {

  if (scores.size() == 1) {
    table_lookup_single_score(table_storage, dtypes, bucket_capacity, keys,
                              scores, policies, is_returns, founds, values);
  } else {
    throw std::runtime_error("Not support multi-scores.");
  }
}

} // namespace dyn_emb