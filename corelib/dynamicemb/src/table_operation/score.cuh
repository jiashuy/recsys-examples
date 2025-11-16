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

#include <cstdint>

#include <vector>

#include <cuda_runtime.h>

namespace dyn_emb {

enum class ScorePolicyType : uint32_t {
  Const = 0,
  Assign = 1,
  Accumulate = 2,
  GlobalTimer = 3,
};

struct ScoreCoder {

  constexpr int MaxScoreNumber = sizeof(uint64_t);

  static __host__ void encode(uint64_t &code, int index, bool is_return,
                              ScorePolicyType policy_type) {

    if (index < 0 or index >= MaxScoreNumber) {
      throw std::invalid_argument("Index overflow.");
    }
    uint8_t score_code = static_cast<uint8_t>(policy_type);
    if (is_return) {
      score_code |= 0x80;
    }

    code &= ~(uint64_t(0xFF) << (index * 8));
    code |= (uint64_t(score_code) << (index * 8));

    return;
  }

  static __device__ __forceinline__ bool decode(uint64_t code, int index,
                                                ScorePolicyType &policy_type) {
    uint8_t score_code = (code >> (index * 8)) & 0xFF;
    bool is_return = (score_code & 0x80) != 0;
    score_code &= 0x7F;
    policy_type = static_cast<ScorePolicyType>(score_code);
    return is_return;
  }
};

uint64_t get_scores_code(std::vector<ScorePolicyType> types,
                         std::vector<bool> is_returns) {
  if (types.size() == 0 or types.size() != is_returns.size()) {
    throw std::invalid_argument("Empty argument or size mismatch.");
  }

  int max_score_num = ScoreCoder::MaxScoreNumber;
  if (max_score_num < types.size()) {
    throw std::runtime_error("Can't convert as overflow.");
  }

  uint64_t code = 0;

  for (int64_t i = 0; i < types.size(); i++) {
    ScoreCoder::encode(code, i, is_returns[i], types[i]);
  }

  return code;
}

template <typename... Ts> struct ScoreTuple;

template <typename T> struct ScoreTuple<T> {
  static int64_t size() { return sizeof(T); }

  T value;
};

template <typename T1, typename T2> struct ScoreTuple<T1, T2> {
  static int64_t size() { return sizeof(T1) + sizeof(T2); }

  T1 value1;
  T2 value2;
};

// template <typename ScorePolicyImpl>
// struct ScorePolicy {

//   using ScoreTuple = typename ScorePolicyImpl::ScoreTuple;
//   using ScoreArgument = typename ScorePolicyImpl::ScoreArgument;

//   static __device__ __forceinline__ void get(ScoreArgument& arg, int64_t
//   index, ScoreTuple& score) {
//     static_cast<ScorePolicyImpl*>(this)->get(arg, index, score);
//   }

//   template<typename BucketType>
//   static __device__ __forceinline__ void update(BucketType& bucket, typename
//   BucketType::Iterator iter, ScoreTuple& score, uint64_t score_code) {
//     static_cast<ScorePolicyImpl*>(this)->update(bucket, iter, score,
//     score_code);
//   }

//   static __device__ __forceinline__ void set(ScoreArgument& arg, int64_t
//   index, ScoreTuple score) {
//     static_cast<ScorePolicyImpl*>(this)->set(arg, index, score);
//   }
// };

template <typename... Ts> struct ScorePolicy;

template <typename T> struct ScorePolicy<T> {

  using ScoreTuple = ScoreTuple<T>;

  struct ScoreArgument {
    T *__restrict__ data;
    uint64_t code;
  };

  static __device__ __forceinline__ void get(ScoreArgument &arg, int64_t index,
                                             ScoreTuple &score) {
    ScorePolicyType policy_type;
    ScoreCoder::decode(arg.code, 0, policy_type);

    if (policy_type == ScorePolicyType::Const) {
      return;
    }
    if (policy_type == ScorePolicyType::GlobalTimer) {
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(score.value));
    } else {
      score.value = arg.data[index];
    }
  }

  template <typename BucketType>
  static __device__ __forceinline__ void
  update(BucketType &bucket, typename BucketType::Iterator iter,
         ScoreTuple &score, uint64_t score_code) {

    ScorePolicyType policy_type;
    bool is_return = ScoreCoder::decode(score_code, 0, policy_type);

    if (policy_type == ScorePolicyType::Const) {
      if (is_return) {
        score.value = *bucket.scores<0>(iter);
      }
      return;
    }
    if (policy_type == ScorePolicyType::Accumulate) {
      score.value += *bucket.scores<0>(iter);
      *bucket.scores<0>(iter) = score.value;
    } else {
      *bucket.scores<0>(iter) = score.value;
    }
  }

  static __device__ __forceinline__ void set(ScoreArgument &arg, int64_t index,
                                             ScoreTuple score) {
    ScorePolicyType policy_type;
    bool is_return = ScoreCoder::decode(arg.code, 0, policy_type);
    if (is_return) {
      arg.data[index] = score.value;
    }
  }
};

template <typename T1, typename T2> struct ScorePolicy<T1, T2> {

  using ScoreTuple = ScoreTuple<T1, T2>;

  struct ScoreArgument {
    T1 *__restrict__ data1;
    T2 *__restrict__ data2;
    uint64_t code;
  };

  static __device__ __forceinline__ void get(ScoreArgument &arg, int64_t index,
                                             ScoreTuple &score) {
    ScorePolicyType policy_type1, policy_type2;
    ScoreCoder::decode(arg.code, 0, policy_type1);
    ScoreCoder::decode(arg.code, 1, policy_type2);

    if (policy_type == ScorePolicyType::Const) {
      return;
    }

    if (policy_type1 == ScorePolicyType::Const) {
    } else if (policy_type1 == ScorePolicyType::GlobalTimer) {
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(score.value1));
    } else {
      score.value1 = arg.data1[index];
    }

    if (policy_type2 == ScorePolicyType::Const) {
      return;
    }
    if (policy_type2 == ScorePolicyType::GlobalTimer) {
      asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(score.value2));
    } else {
      score.value2 = arg.data2[index];
    }
  }

  template <typename BucketType>
  static __device__ __forceinline__ void
  update(BucketType &bucket, typename BucketType::Iterator iter,
         ScoreTuple &score, uint64_t score_code) {

    ScorePolicyType policy_type1, policy_type2;
    bool is_return1 = ScoreCoder::decode(score_code, 0, policy_type1);
    bool is_return2 = ScoreCoder::decode(score_code, 1, policy_type2);

    if (policy_type1 == ScorePolicyType::Const) {
      if (is_return1) {
        score.value1 = *bucket.scores<0>(iter);
      }
    } else if (policy_type1 == ScorePolicyType::Accumulate) {
      score.value1 += *bucket.scores<0>(iter);
      *bucket.scores<0>(iter) = score.value1;
    } else {
      *bucket.scores<0>(iter) = score.value1;
    }

    if (policy_type2 == ScorePolicyType::Const) {
      if (is_return2) {
        score.value1 = *bucket.scores<1>(iter);
      }
      return;
    }
    if (policy_type2 == ScorePolicyType::Accumulate) {
      score.value2 += *bucket.scores<1>(iter);
      *bucket.scores<1>(iter) = score.value2;
    } else {
      *bucket.scores<1>(iter) = score.value2;
    }
  }

  static __device__ __forceinline__ void set(ScoreArgument &arg, int64_t index,
                                             ScoreTuple score) {
    ScorePolicyType policy_type1, policy_type2;
    bool is_return1 = ScoreCoder::decode(score_code, 0, policy_type1);
    bool is_return2 = ScoreCoder::decode(score_code, 1, policy_type2);
    if (is_return1) {
      arg.data1[index] = score.value1;
    }
    if (is_return2) {
      arg.data2[index] = score.value2;
    }
  }
};

template <typename ScoreImpl> struct ScoreBase {

  using ScoreTuple = typename ScoreImpl::ScoreTuple;

  int64_t size() const { static_cast<ScoreImpl *>(this)->size(); }

  __forceinline__ __device__ ScoreTuple load(uint8_t const *storage,
                                             int64_t index) const {
    static_cast<ScoreImpl *>(this)->load(storage, index);
  }

  template <typename BucketType, typename ReductionTraits>
  __forceinline__ __device__ void
  reduce(BucketType &bucket, ScoreTuple &min_score,
         typename BucketType::Iterator &min_iter,
         typename BucketType::KeyType &min_key, uint8_t *sm_buf) {
    static_cast<ScoreImpl *>(this)->reduce(bucket, min_score, min_iter, min_key,
                                           sm_buf);
  }
};

template <typename T> struct SingleScore : ScoreBase<SingleScore> {

  using ScoreTuple = T;

  int64_t size() const { return sizeof(T); }

  __forceinline__ __device__ ScoreTuple load(uint8_t const *storage,
                                             int64_t index) const {
    return *(reinterpret_cast<const T *>(storage) + index);
  }

  template <typename BucketType, typename ReductionTraits>
  __forceinline__ __device__ void
  reduce(BucketType &bucket, ScoreTuple &min_score,
         typename BucketType::Iterator &min_iter,
         typename BucketType::KeyType &min_key, uint8_t *sm_buf) {}
};

template <typename T1, typename T2>
struct DoubleScore : ScoreBase<DoubleScore> {

  struct ScoreTuple {
    T1 score1;
    T2 score2;
  };

  int64_t size() const { return sizeof(T1) + sizeof(T2); }

  __forceinline__ __device__ ScoreTuple load(uint8_t const *storage,
                                             int64_t index) const {
    ScoreTuple result;
    result.score1 = *(reinterpret_cast<const T *>(storage) + index);
  }

  template <typename BucketType, typename ReductionTraits>
  __forceinline__ __device__ void
  reduce(BucketType &bucket, ScoreTuple &min_score,
         typename BucketType::Iterator &min_iter,
         typename BucketType::KeyType &min_key, uint8_t *sm_buf) {}
};

} // namespace dyn_emb