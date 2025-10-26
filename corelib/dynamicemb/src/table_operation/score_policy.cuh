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

#include <cuda_runtime.h>

namespace dyn_emb {

template <typename ScoreType> struct ScoreArgument {
  ScoreType *__restrict__ scores;
};

template <typename ScoreType, bool ReturnScore> struct ReadScorePolicy {
  using Argument = ScoreArgument<ScoreType>;

  static constexpr bool ReadOnly = true;

  static __device__ __forceinline__ ScoreType get(Argument arg, int64_t index) {
    return ScoreType();
  }

  static __device__ __forceinline__ ScoreType update(ScoreType *dst,
                                                     ScoreType score) {
    return *dst;
  }

  static __device__ __forceinline__ void set(Argument arg, int64_t index,
                                             ScoreType score) {
    if constexpr (ReturnScore) {
      arg.scores[index] = score;
    }
  }
};

template <typename ScoreType, bool ReturnScore> struct LruScorePolicy {

  using Argument = ScoreArgument<ScoreType>;

  static constexpr bool ReadOnly = false;

  static __device__ __forceinline__ ScoreType get(Argument arg, int64_t index) {
    ScoreType mclk;
    asm volatile("mov.u64 %0,%%globaltimer;" : "=l"(mclk));
    return mclk;
  }

  static __device__ __forceinline__ ScoreType update(ScoreType *dst,
                                                     ScoreType score) {
    *dst = score;
    return score;
  }

  static __device__ __forceinline__ void set(Argument arg, int64_t index,
                                             ScoreType score) {
    if constexpr (ReturnScore) {
      arg.scores[index] = score;
    }
  }
};

template <typename ScoreType, bool ReturnScore> struct LfuScorePolicy {

  static constexpr bool ReadOnly = false;

  using Argument = ScoreArgument<ScoreType>;

  static __device__ __forceinline__ ScoreType get(Argument arg, int64_t index) {
    return arg.scores[index];
  }

  static __device__ __forceinline__ ScoreType update(ScoreType *dst,
                                                     ScoreType score) {
    // if constexpr (AccumulateScore) {
    ScoreType score_new = *dst + score;
    *dst = score_new;
    return score_new;
    // } else {
    //   *dst = score;
    //   return score;
    // }
  }

  static __device__ __forceinline__ void set(Argument arg, int64_t index,
                                             ScoreType score) {
    if constexpr (ReturnScore) {
      arg.scores[index] = score;
    }
  }
};

template <typename ScoreType, bool ReturnScore> struct CustomizedScorePolicy {

  static constexpr bool ReadOnly = false;

  using Argument = ScoreArgument<ScoreType>;

  static __device__ __forceinline__ ScoreType get(Argument arg, int64_t index) {
    return arg.scores[index];
  }

  static __device__ __forceinline__ ScoreType update(ScoreType *dst,
                                                     ScoreType score) {
    *dst = score;
    return score;
  }

  static __device__ __forceinline__ void set(Argument arg, int64_t index,
                                             ScoreType score) {
    if constexpr (ReturnScore) {
      arg.scores[index] = score;
    }
  }
};

} // namespace dyn_emb