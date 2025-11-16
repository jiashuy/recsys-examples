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
#include <stddef.h>
#include <type_traits>
#include <utility>

#include <cuda/atomic>
#include <cuda/std/semaphore>
#include <cuda_runtime.h>

#include "score.cuh"

namespace dyn_emb {

using CounterType = int64_t;
using ScoreType = uint64_t;
using DigestType = uint8_t;
using IndexType = int64_t;

__device__ __forceinline__ CounterType atomicAdd(CounterType *address,
                                                 const CounterType val) {
  return (CounterType)atomicAdd((unsigned long long *)address, val);
}

enum class InsertResult : uint8_t {
  Insert,     // Insert into an empty or reclaimed slot.
  Assign,     // Hit and assign.
  Evict,      // Evict a key and insert into the evicted slot.
  Duplicated, // Meet duplicated keys on the fly.
  Busy,       // Insert failed as all slots busy.
  Init,
};

__device__ __forceinline__ bool isInsertSuccess(InsertResult result) {
  if (static_cast<uint8_t>(result) <=
      static_cast<uint8_t>(InsertResult::Evict)) {
    return true;
  }
  return false;
}

enum class EvictPolicy : int {
  kLru = 0,
  kLfu = 1,
  kCustomized = 2,
};

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

// Select from double buffer.
// If i % 2 == 0, select buffer 0, else buffer 1.
__forceinline__ __device__ int same_buf(int i) { return (i & 0x01) ^ 0; }
// If i % 2 == 0, select buffer 1, else buffer 0.
__forceinline__ __device__ int diff_buf(int i) { return (i & 0x01) ^ 1; }

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
struct LinearBucket {

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

  __device__ __forceinline__ bool try_lock(Iterator &iter, KeyType &key) {
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
      typename BucketTraits::DigestVectorType; // used for comparison
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
  using ScoreVector = typename BucketTraits::ScoreVectorType;
  // static constexpr int NumScorePerVector = BucketTraits::NumScorePerVector;

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

  __forceinline__ __device__ int capacity() const { return capacity_; }

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

  template <typename T, int N, int Stride,
            typename = std::enable_if_t<sizeof(T) * Stride <= 16>>
  __forceinline__ __device__ void async_copy_bulk(T *dst, T const *src) {
    static_assert(N % Stride == 0);
#pragma unroll
    for (int i = 0; i < N; i += Stride) {
      __pipeline_memcpy_async(dst + i, src + i, sizeof(T) * Stride);
    }
  }

  /*
  Let iter and step have a state, and if they have been probed, they will not be
  probed again
  */
  template <int GroupSize = 1>
  __forceinline__ __device__ bool probe(KeyType key, Iterator &iter,
                                        int &step) const {

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
  __forceinline__ __device__ void reduce(Iterator &min_iter,
                                         ScoreType &min_score, KeyType &dst_key,
                                         ScoreType *sm_buffers) const {

    static constexpr int BulkDim = BufferDim / 2;
    static constexpr int BlockSize = KernelTraits::ThreadBlockDim;
    static constexpr int GroupSize = KernelTraits::ReductionGroupSize;
    static constexpr bool PipelinedReduction = KernelTraits::PipelinedReduction;
    static constexpr int BufferDim = KernelTraits::NumScorePerThreadBuffer;
    static constexpr int BulkDim =
        PipelinedReduction ? (BufferDim / 2) : BufferDim;

    static constexpr int Stride = sizeof(ScoreVector);

    /// TODO: support more
    static_assert(PipelinedReduction == true);
    static_assert(GroupSize == 1);

    Iterator iter = 0;
    min_score = MaxScore;

    int rank = threadIdx.x;

    async_copy_bulk<ScoreType, BulkDim, Stride>(sm_buffers + rank * BufferDim,
                                                scores(iter));
    __pipeline_commit();

    for (; iter < capacity_; iter += BulkDim) {
      if (i < capacity_ - BulkDim) {
        async_copy_bulk<ScoreType, BulkDim, Stride>(
            sm_buffers + rank * BufferDim + diff_buf(iter / BulkDim) * BulkDim,
            scores(iter) + BulkDim);
      }
      __pipeline_commit();
      __pipeline_wait_prior(1);
      ScoreType temp_scores[Stride];
      ScoreType *src =
          sm_buffers + rank * BufferDim + same_buf(iter / BulkDim) * BulkDim;
#pragma unroll
      for (int k = 0; k < BulkDim; k += Stride) {
        *reinterpret_cast<ScoreVector *>(temp_scores) =
            *reinterpret_cast<ScoreVector *>(src + k);
#pragma unroll
        for (int j = 0; j < Stride; j += 1) {
          ScoreType temp_score = temp_scores[j];
          if (temp_score < min_score) {
            auto temp_key_slot =
                reinterpret_cast<AtomicKey *>(keys(iter + k + j));

            auto temp_key =
                temp_key_slot->load(cuda::std::memory_order_relaxed);

            if (temp_key != LockedKey && temp_key != EmptyKey) {
              min_score = temp_score;
              min_iter = iter + k + j;
              dst_key = temp_key;
            }
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

  BucketHashingTable(uint8_t *storage, int64_t num_buckets,
                     int64_t bucket_capacity)
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

  __device__ __forceinline__ BucketType get_bucket(KeyType key) const {
    auto hashcode = hash(key);
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
  int64_t bucket_capacity_;
};

} // namespace dyn_emb