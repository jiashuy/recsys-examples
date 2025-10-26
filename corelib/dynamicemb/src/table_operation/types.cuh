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

#include <array>
#include <cstdint>
#include <cuda/atomic>
#include <cuda/std/semaphore>
#include <cuda_runtime.h>
#include <stddef.h>
#include <type_traits>
#include <utility>

#include "score_policy.cuh"

namespace dyn_emb {

template <std::size_t Index, typename... Types> struct OffsetHelper;

template <typename First, typename... Rest>
struct OffsetHelper<0, First, Rest...> {
  static constexpr std::size_t offset = 0;
};

template <std::size_t Index, typename First, typename... Rest>
struct OffsetHelper<Index, First, Rest...> {
  static constexpr std::size_t offset =
      sizeof(First) + OffsetHelper<Index - 1, Rest...>::offset;
};

template <typename... Types> struct TypeInfo {
  static constexpr std::size_t total_size = (0 + ... + sizeof(Types));

  static constexpr std::size_t type_count = sizeof...(Types);

  template <std::size_t Index> static constexpr std::size_t get_offset() {
    static_assert(Index < type_count, "Index out of range");
    return OffsetHelper<Index, Types...>::offset;
  }
};

using ScoreType = uint64_t;
using DigestType = uint8_t;

template <typename DigestType,
          uint64_t EMPTY_KEY = UINT64_C(0xFFFFFFFFFFFFFFFF),
          uint64_t LOCKED_KEY = UINT64_C(0xFFFFFFFFFFFFFFFD),
          uint64_t RECLAIM_KEY = UINT64_C(0xFFFFFFFFFFFFFFFE),
          uint64_t RESERVE_KEY_MASK = UINT64_C(0xFFFFFFFFFFFFFFFC),
          uint64_t EMPTY_SCORE = UINT64_C(0),
          uint64_t MAX_SCORE = UINT64_C(0xFFFFFFFFFFFFFFFF)>
struct SlotsBucketTraits {

  static constexpr uint64_t EmptyKey = EMPTY_KEY;
  static constexpr uint64_t LockedKey = LOCKED_KEY;
  static constexpr uint64_t ReclaimKey = RECLAIM_KEY;

  static constexpr uint64_t EmptyScore = EMPTY_SCORE;
  static constexpr uint64_t MaxScore = MAX_SCORE;

  static constexpr uint64_t ReserveKeyMask = RESERVE_KEY_MASK;

  using DigestVectorType = uint32_t;

  using DigestBufferType = uint4;

  using ComparedResult = int;

  static constexpr int VectorDim =
      sizeof(DigestVectorType) / sizeof(DigestType);
  static constexpr int BufferDim =
      sizeof(DigestBufferType) / sizeof(DigestType);
  static constexpr int NumVectorPerBuffer =
      sizeof(DigestBufferType) / sizeof(DigestVectorType);

  struct DigestVectorComparator {
    static __device__ __forceinline__ ComparedResult
    compare(DigestVectorType lhs, DigestVectorType rhs) {
      // Perform a vectorized comparison by byte,
      // and if they are equal, set the corresponding byte in the result to
      // 0xff.
      ComparedResult cmp_result = __vcmpeq4(lhs, rhs);
      cmp_result &= 0x01010101;
      return cmp_result;
    }

    static __device__ __forceinline__ int
    equal_index(ComparedResult &cmp_result) {
      if (cmp_result == 0)
        return -1;
      // CUDA uses little endian,
      // and the lowest byte in register stores in the lowest address.
      int index = (__ffs(cmp_result) - 1) >> 3;
      cmp_result &= (cmp_result - 1);
      return index;
    }
  };
};

template <typename KeyType_, typename ValType_, typename ScoreType_,
          typename DigestType, typename BucketTraits,
          typename = std::enable_if_t<std::is_integral_v<KeyType_> &&
                                      sizeof(KeyType_) == 8>>
struct SlotsBucket {

  using KeyType = KeyType_;
  using ValType = ValType_;
  using ScoreType = ScoreType_;

  // using Bucket =
  //     SlotsBucket<KeyType, ValType, ScoreType, DigestType, BucketTraits>;
  /*
  Iterator:
  */
  using Iterator = int;

  template <uint32_t AlignSize>
  static __forceinline__ __device__ int align(Iterator &iter) {
    // iter - (iter % AlignSize)
    constexpr uint32_t MASK = 0xffffffffU - (AlignSize - 1);
    return iter & MASK;
  }

  /*
  Keys:
  */
  using AtomicKey = cuda::atomic<KeyType, cuda::thread_scope_device>;

  static constexpr uint64_t EmptyKey = BucketTraits::EmptyKey;
  static constexpr uint64_t LockedKey = BucketTraits::LockedKey;
  static constexpr uint64_t ReclaimKey = BucketTraits::ReclaimKey;

  static constexpr uint64_t EmptyScore = BucketTraits::EmptyScore;
  static constexpr uint64_t MaxScore = BucketTraits::MaxScore;

  static constexpr uint64_t ReserveKeyMask = BucketTraits::ReserveKeyMask;

  static __device__ __forceinline__ KeyType hash(uint64_t key) {
    uint64_t k = key;
    k ^= k >> 33;
    k *= UINT64_C(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k *= UINT64_C(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    return static_cast<KeyType>(k);
  }

  static __device__ __forceinline__ bool is_valid(uint64_t const &key) {
    return (key & ReserveKeyMask) == ReserveKeyMask;
  }

  __device__ __forceinline__ bool try_lock(Iterator &iter, KeyType& key) {
    auto key_slot = reinterpret_cast<AtomicKey *>(keys(iter));
    return key_slot->compare_exchange_strong(
        key, static_cast<KeyType>(LockedKey), cuda::std::memory_order_acquire,
        cuda::std::memory_order_relaxed);
  }

  __device__ __forceinline__ void unlock(Iterator &iter, KeyType key) {
    auto key_slot = reinterpret_cast<AtomicKey *>(keys(iter));
    key_slot->store(key, cuda::std::memory_order_release);
  }

  __device__ __forceinline__ bool is_empty(Iterator &iter) const {
    auto key_slot = reinterpret_cast<AtomicKey *>(keys(iter));
    auto slot_key = key_slot->load(cuda::std::memory_order_relaxed);
    return slot_key == EmptyKey;
  }

  __device__ __forceinline__ bool is_locked(Iterator &iter) const {
    auto key_slot = reinterpret_cast<AtomicKey *>(keys(iter));
    auto slot_key = key_slot->load(cuda::std::memory_order_relaxed);
    return slot_key == LockedKey;
  }

  /*
  Digest:
  */
  using DigestVector =
      typename BucketTraits::DigestVectorType; // used for comparision
  using DigestBuffer =
      typename BucketTraits::DigestBufferType; // used for loading
  using VectorComparator = typename BucketTraits::DigestVectorComparator;

  static constexpr int VectorDim = BucketTraits::VectorDim;
  static constexpr int BufferDim = BucketTraits::BufferDim;
  static constexpr int NumVectorPerBuffer = BucketTraits::NumVectorPerBuffer;

  static __device__ __forceinline__ DigestType key_to_digest(KeyType key) {
    auto hashcode = hash(key);
    return hashcode_to_digest(hashcode);
  }

  static __device__ __forceinline__ DigestType
  hashcode_to_digest(KeyType hashcode) {
    return static_cast<DigestType>(hashcode >> 32);
  }

  static __device__ __forceinline__ DigestVector
  digest_to_vector(DigestType digest) {
    return static_cast<DigestVector>(__byte_perm(digest, digest, 0x0000));
  }

  static __device__ __forceinline__ void
  digest_buffer_to_vector(DigestBuffer const &digest_buffer,
                          DigestVector digest_vec[NumVectorPerBuffer]) {
    digest_vec[0] = digest_buffer.x;
    digest_vec[1] = digest_buffer.y;
    digest_vec[2] = digest_buffer.z;
    digest_vec[3] = digest_buffer.w;
  }


  /*
  Scores:
  */
 using ScoreVector =
      typename BucketTraits::ScoreVectorType;
  static constexpr int NumScorePerVector = BucketTraits::NumScorePerVector;

  __forceinline__ __device__ SlotsBucket(uint8_t *storage, int capacity)
      : storage_(storage), capacity_(capacity) {}

  __forceinline__ __device__ SlotsBucket() : SlotsBucket(nullptr, 0) {}

  /// @brief TODO:nvcc can't expand it
  // using BucketLayout = TypeInfo<KeyType, ValType, ScoreType, DigestType>;
  static constexpr int KeyOffset = 0;
  static constexpr int ValOffset = KeyOffset + sizeof(KeyType);
  static constexpr int ScoreOffset = ValOffset + sizeof(ValType);
  static constexpr int DigestOffset = ScoreOffset + sizeof(ScoreType);
  static constexpr int BucketBytes = DigestOffset + sizeof(DigestType);

  static __device__ __forceinline__ uint64_t memory_usage(int size) {
    return BucketBytes * size;
  }

  __forceinline__ __device__ int capacity() const {
    return capacity_;
  }

  __forceinline__ __device__ KeyType *keys(const Iterator &iter) const {
    return reinterpret_cast<KeyType *>(storage_ + KeyOffset * capacity_) + iter;
  }

  __forceinline__ __device__ ValType *values(const Iterator &iter) const {
    return reinterpret_cast<ValType *>(storage_ + ValOffset * capacity_) + iter;
  }

  __forceinline__ __device__ ScoreType *scores(const Iterator &iter) const {
    return reinterpret_cast<ScoreType *>(storage_ + ScoreOffset * capacity_) +
           iter;
  }

  __forceinline__ __device__ DigestType *digests(const Iterator &iter) const {
    return reinterpret_cast<DigestType *>(storage_ + DigestOffset * capacity_) +
           iter;
  }

  enum class ProbeResult : uint8_t {
    Init = 0,
    Existed = 1,
    Empty = 2,
    Exhausted = 3,
    Failed = 4,
    Absent = 5,
  };


  /*
  Let iter and step have a state, and if they have been probed, they will not be probed again
  */
  template <int GroupSize = 1>
  __forceinline__ __device__ bool probe(KeyType key, Iterator &iter, int& step) const {

    if (step == capacity_) {
      return ProbeResult::Exhausted;
    }

    auto hashcode = hash(key);
    auto digest = hashcode_to_digest(hashcode);
    auto digest_vec = digest_to_vector(digest);

    /// TODO: support more
    static_assert(GroupSize == 1);
    // bool early_stop = false; // used when GroupSize > 1
    if (storage_ == nullptr or capacity_ == 0) {
      // early_stop = true;
      return ProbeResult::Failed;
    }

    if (iter < 0 or iter > capacity_) {
      iter = hashcode % capacity_;
    }
    constexpr int Stride = BufferDim;

    iter = align<Stride>(iter);

    auto empty_digest = key_to_digest(EmptyKey);
    auto empty_vec = digest_to_vector(empty_digest);

    ProbeResult result = ProbeResult::Init;

    for (; step < capacity_; step += Stride) {

      iter = (iter + step) % capacity_;

      auto buffer = *(reinterpret_cast<DigestBuffer *>(digests(iter)));

      constexpr int Length = NumVectorPerBuffer;

      DigestVector vec[Length];
      digest_buffer_to_vector(buffer, vec);

      for (int i = 0; i < Length; i++) {

        auto cmp_res = VectorComparator::compare(vec[i], digest_vec);
        while (true) {
          int offset = VectorComparator::equal_index(cmp_res);
          if (offset < 0)
            break;

          auto possible_iter = iter + i * VectorDim + offset;

          auto possible_key_slot =
              reinterpret_cast<AtomicKey *>(keys(possible_iter));

          auto possible_key =
              possible_key_slot->load(cuda::std::memory_order_relaxed);

          if (possible_key == key) {
            iter = possible_iter;
            return ProbeResult::Existed;
          }
        }
        cmp_res = VectorComparator::compare(vec[i], empty_vec);
        while (true) {
          int offset = VectorComparator::equal_index(cmp_res);
          if (offset < 0)
            break;

          auto possible_iter = iter + i * VectorDim + offset;

          auto possible_key_slot =
              reinterpret_cast<AtomicKey *>(keys(possible_iter));

          auto possible_key =
              possible_key_slot->load(cuda::std::memory_order_relaxed);

          if (possible_key == EmptyKey) {
            iter = possible_iter;
            return ProbeResult::Empty;
          }
        }
      }
    }
    return ProbeResult::Exhausted;
  }

  template <typename KernelTraits>
  __forceinline__ __device__ void reduce(Iterator &iter, KeyType& dst_key, ScoreType src_score, ScoreType* sm_buffers) const {
    static constexpr int BlockSize = KernelTraits::ThreadBlockDim;
    static constexpr int GroupSize = KernelTraits::ReductionGroupSize;
    static constexpr bool PipelinedReduction = KernelTraits::PipelinedReduction;
    static constexpr int NumScorePerBuffer =
        PipelinedReduction ? (KernelTraits::NumScorePerThreadBuffer / 2)
                          : KernelTraits::NumScorePerThreadBuffer;
    
    static_assert(PipelinedReduction == true);

    iter = -1;
    ScoreType min_score = MaxScore;

    // constexpr uint32_t STRIDE_S = 4;
    // constexpr uint32_t Load_LEN_S = sizeof(byte16) / sizeof(S);

  #pragma unroll
      for (int j = 0; j < NumScorePerBuffer; j += NumScorePerVector) {
        __pipeline_memcpy_async(sm_bucket_scores[tx] + j, bucket_scores_ptr + j,
                                sizeof(S) * Load_LEN_S);
      }
      __pipeline_commit();
      for (int i = 0; i < bucket_capacity; i += STRIDE_S) {
        if (i < bucket_capacity - STRIDE_S) {
  #pragma unroll
          for (int j = 0; j < STRIDE_S; j += Load_LEN_S) {
            __pipeline_memcpy_async(
                sm_bucket_scores[tx] + diff_buf(i / STRIDE_S) * STRIDE_S + j,
                bucket_scores_ptr + i + STRIDE_S + j, sizeof(S) * Load_LEN_S);
          }
        }
        __pipeline_commit();
        __pipeline_wait_prior(1);
        S temp_scores[Load_LEN_S];
        S* src = sm_bucket_scores[tx] + same_buf(i / STRIDE_S) * STRIDE_S;
  #pragma unroll
        for (int k = 0; k < STRIDE_S; k += Load_LEN_S) {
          *reinterpret_cast<byte16*>(temp_scores) =
              *reinterpret_cast<byte16*>(src + k);
  #pragma unroll
          for (int j = 0; j < Load_LEN_S; j += 1) {
            S temp_score = temp_scores[j];
            if (temp_score < min_score) {
              auto verify_key_ptr = BUCKET::keys(bucket_keys_ptr, i + k + j);
              auto verify_key =
                  verify_key_ptr->load(cuda::std::memory_order_relaxed);
              if (verify_key != static_cast<K>(LOCKED_KEY) &&
                  verify_key != static_cast<K>(EMPTY_KEY)) {
                min_score = temp_score;
                min_pos = i + k + j;
              }
            }
          }
        }
      }


    while (occupy_result == OccupyResult::INITIAL) {
      S* bucket_scores_ptr = BUCKET::scores(bucket_keys_ptr, bucket_capacity, 0);
      S min_score = MAX_SCORE;
      int min_pos = -1;

      score = ScoreFunctor::desired_when_missed(scores, kv_idx, global_epoch);
      if (score <= min_score) {
        occupy_result = OccupyResult::REFUSED;
        break;
      }
      auto min_score_key = BUCKET::keys(bucket_keys_ptr, min_pos);
      auto expected_key = min_score_key->load(cuda::std::memory_order_relaxed);
      if (expected_key != static_cast<K>(LOCKED_KEY) &&
          expected_key != static_cast<K>(EMPTY_KEY)) {
        bool result = min_score_key->compare_exchange_strong(
            expected_key, static_cast<K>(LOCKED_KEY),
            cuda::std::memory_order_acquire, cuda::std::memory_order_relaxed);
        if (result) {
          S* min_score_ptr =
              BUCKET::scores(bucket_keys_ptr, bucket_capacity, min_pos);
          auto verify_score_ptr =
              reinterpret_cast<AtomicScore<S>*>(min_score_ptr);
          auto verify_score =
              verify_score_ptr->load(cuda::std::memory_order_relaxed);
          if (verify_score <= min_score) {
            key_pos = min_pos;
            ScoreFunctor::update_with_digest(
                bucket_keys_ptr, key_pos, scores, kv_idx, score, bucket_capacity,
                get_digest<K>(key), (occupy_result != OccupyResult::DUPLICATE));
            if (expected_key == static_cast<K>(RECLAIM_KEY)) {
              occupy_result = OccupyResult::OCCUPIED_RECLAIMED;
              atomicAdd(bucket_size_ptr, 1);
            } else {
              occupy_result = OccupyResult::EVICT;
            }
          } else {
            min_score_key->store(expected_key, cuda::std::memory_order_release);
          }
        }
      }
    }
  }

  uint8_t *__restrict__ storage_;
  int capacity_;
};

template <typename BucketType_> struct BucketHashingTable {
  using BucketType = BucketType_;
  using KeyType = typename BucketType::KeyType;
  using ScoreType = typename BucketType::ScoreType;
  using ValType = typename BucketType::ValType;

  BucketHashingTable(uint8_t *storage, int64_t num_buckets, int bucket_capacity)
      : storage_(storage), num_buckets_(num_buckets),
        bucket_capacity_(bucket_capacity) {}

  static __device__ __forceinline__ KeyType hash(uint64_t key) {
    return BucketType::hash(key);
  }

  __device__ __forceinline__ BucketType operator[](uint64_t idx) const {
    // assert(idx >= num_buckets_);
    auto bucket_raw_data =
        // storage_ + BucketType::total_size * bucket_capacity_ * idx;
        storage_ + BucketType::memory_usage(bucket_capacity_) * idx;
    return BucketType(bucket_raw_data, bucket_capacity_);
  }

  __device__ __forceinline__ uint64_t capacity() const {
    return num_buckets_ * bucket_capacity_;
  }

  __device__ __forceinline__ int bucket_capacity() const {
    return bucket_capacity_;
  }

  uint8_t *__restrict__ storage_;
  int64_t num_buckets_;
  int bucket_capacity_;
};

template <int ProbingGroupSize_, int ReductionGroupSize_> struct KernelTraits {
  static constexpr int ProbingGroupSize = ProbingGroupSize_;
  static constexpr int ReductionGroupSize = ReductionGroupSize_;
};

enum class EvictPolicy : int {
  kLru = 0,
  kLfu = 1,
  kCustomized = 2,
};

} // namespace dyn_emb