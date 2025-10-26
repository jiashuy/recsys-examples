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

#pragma once

#include "types.cuh"
#include <cub/cub.cuh>

namespace dyn_emb {

template <typename Table, typename ScorePolicy, typename KernelTraits>
__global__ void
table_lookup_kernel(Table table, int64_t batch,
                    typename Table::KeyType const *__restrict__ input_keys,
                    typename Table::ValType *__restrict__ output_values,
                    ScoreArgument<typename Table::ScoreType> score_args,
                    bool *__restrict__ founds) {

  using KeyType = typename Table::KeyType;
  using ValType = typename Table::ValType;
  using ScoreType = typename Table::ScoreType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = Bucket::EmptyKey;
    ScoreType score = Bucket::EmptyScore;

    key = input_keys[i];
    score = ScorePolicy::get(score_args, i);

    Bucket bucket = Bucket();
    KeyType hashcode = KeyType();
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      auto bucket_id = (hashcode % table.capacity()) / table.bucket_capacity();
      bucket = table[bucket_id];
    }
    Iter iter = Iter(hashcode % table.bucket_capacity());
    int step = 0;
    auto probe_res = bucket.probe<KernelTraits::ProbingGroupSize>(key, iter, step);
    bool found = probe_res == Bucket::ProbeResult::Existed;
    ValType val = ValType();
    if (found) {

      if constexpr (ScorePolicy::ReadOnly) {
        score = *bucket.scores(iter);
      } else {
        KeyType expected_key = key;
        if (bucket.try_lock(iter, expected_key)) {
          score = ScorePolicy::update(bucket.scores(iter), score);
          bucket.unlock(iter, key);
        } else {
          found = false; // only one update will succeed for duplicated keys.
          score = Bucket::EmptyScore;
        }
      }

      // read the values if needed.
      if (found and output_values) {
        val = *bucket.values(iter);
      }
    }
    if (output_values) {
      output_values[i] = val;
    }
    ScorePolicy::set(score_args, i, score);
    if (founds) {
      founds[i] = found;
    }
  }
}

template <typename Table, typename ScorePolicy, typename KernelTraits>
__global__ void
table_insert_kernel(Table table, int * __restrict__ bucket_sizes, int64_t batch,
                    typename Table::KeyType const *__restrict__ input_keys,
                    typename Table::ValType *__restrict__ input_values,
                    ScoreArgument<typename Table::ScoreType> score_args,
                    typename Table::KeyType ** __restrict__ locked_key_slots,
                    typename Table::ScoreType ** __restrict__ locked_score_slots,
                    typename Table::ScoreType * __restrict__ output_scores,
                    bool value_reuse) {

  using KeyType = typename Table::KeyType;
  using ValType = typename Table::ValType;
  using ScoreType = typename Table::ScoreType;
  using Bucket = typename Table::BucketType;
  using Iter = typename Bucket::Iterator;
  using ProbeResult = typename Bucket::ProbeResult;

  auto tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  extern __shared__ ScoreType sm_score_buffers[];

  for (int64_t i = tid; i < batch; i += gridDim.x * blockDim.x) {

    KeyType key = Bucket::EmptyKey;
    ScoreType score = Bucket::EmptyScore;
    // ValType val = value_reuse ? ValType() : input_values[i];

    key = input_keys[i];
    score = ScorePolicy::get(score_args, i);
    

    Bucket bucket = Bucket();
    KeyType hashcode = KeyType();
    if (Bucket::is_valid(key)) {
      hashcode = Table::hash(key);
      auto bucket_id = (hashcode % table.capacity()) / table.bucket_capacity();
      bucket = table[bucket_id];
      bucket_sizes = bucket_sizes + bucket_id; // don't use bucket_id anymore
    }
    Iter iter = Iter(hashcode % table.bucket_capacity());
    ProbeResult probe_res = ProbeResult::Init;
    int step = 0;
    while (step != bucket.capacity()) {
      probe_res = bucket.probe<KernelTraits::ProbingGroupSize>(key, iter, step);
      if (probe_res == ProbeResult::Existed) {
        KeyType expected_key = key;
        if (bucket.try_lock(iter, expected_key)) {
          // bucket.unlock(iter, key); // will not unlock, to avoid 2 threads got the same slot.
        } else {  // else: the key is evicted from the bucket(full), try to reintert by eviction including reclaimed key.
          probe_res = ProbeResult::Absent;
        }
        break;
      }
      if (probe_res == ProbeResult::Empty) {
        KeyType expected_key = Bucket::EmptyKey;
        if (bucket.try_lock(iter, expected_key)) {
          atomicAdd(bucket_size, 1);
          *bucket.digests(iter) = Bucket::key_to_digest(key);
          probe_res = ProbeResult::Existed;
          break;
        }
      } // else: ProbeResult::Exhausted
    }
    
    while (probe_res != ProbeResult::Existed and probe_res != ProbeResult::Failed) {
      ScoreType compare_score = ScorePolicy::score_for_compare(score);///TODO:score_for_compare
      KeyType evict_key = Bucket::EmptyKey;
      probe_res = bucket.reduce<KernelTraits>(iter, evict_key, compare_score, sm_score_buffers);
      if (probe_res == ProbeResult::Failed) break;
      if (evict_key != Bucket::LockedKey) {
        KeyType expected_key = evict_key;
        if (bucket.try_lock(iter, expected_key)) {
          *bucket.digests(iter) = Bucket::key_to_digest(key);
          if (evict_key == Bucket::ReclaimKey) {
            atomicAdd(bucket_size, 1);
          }
          probe_res = ProbeResult::Existed;
          break;
        }
      }
    }
    if (probe_res == ProbeResult::Existed) {
      score = ScorePolicy::update(bucket.scores(iter), score);
      *bucket.score(iter) = Bucket::MaxScore; // to avoid dead lock when reduce brings by small score
      ScorePolicy::set(score_args, i, score);
      if (output_scores) {
        output_scores[i] = score;
      }
      if (value_reuse) {
        input_values[i] = *bucket.values(iter);
      } else {
        *bucket.values(iter) = input_values[i];
      }
      locked_key_slots[i] = bucket.keys(iter);
      locked_score_slots[i] = bucket.score(iter);
    }
  }
}

} // namespace dyn_emb