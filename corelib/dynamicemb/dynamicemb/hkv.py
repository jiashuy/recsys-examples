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

import warnings
from typing import List, Union
from itertools import accumulate

import torch

from dynamicemb.dynamicemb_config import _next_power_of_2, dtype_to_bytes
from dynamicemb.types import BucketType, MemoryType
from dynamicemb_extensions import(
    table_lookup,
    table_insert,
    table_insert_and_evict,
    tensor_partition,
    table_partition,
)

def uint64_to_int64(x):
    return x if x < (1 << 63) else x - (1 << 64)

def murmur3_hash_64bits(key: int) -> int:
    """
    """
    k = key & 0xFFFFFFFFFFFFFFFF
    
    k ^= k >> 33
    k = (k * 0xff51afd7ed558ccd) & 0xFFFFFFFFFFFFFFFF 
    
    k ^= k >> 33
    k = (k * 0xc4ceb9fe1a85ec53) & 0xFFFFFFFFFFFFFFFF
    
    k ^= k >> 33
    
    return k



class HKV:
    def __init__(
        self,
        key_type: torch.dtype,
        value_type: torch.dtype,
        capacity: int,
        bucket_capacity: int,
        device: torch.device=None,
        value_init_fn=None,
    ):
        self.key_type = key_type
        self.value_type = value_type
        self.device = device if device is None else torch.device("cuda", torch.cuda.current_device())
        accepted_key_types = {torch.int64, torch.uint64}
        assert key_type in accepted_key_types, "Only accept 64 bits integer as key's type."
        self.score_type = torch.uint64
        self.digest_type = torch.uint8
        fileds_type = [self.key_type, self.value_type, self.score_type, self.digest_type]
        fields_byte = [dtype_to_bytes(x) for x in fileds_type]

        max_load_bytes = 16
        assert capacity > 0 and bucket_capacity > 0 and capacity > bucket_capacity
        digest_load_dim = max_load_bytes // fields_byte[-1]
        if bucket_capacity % digest_load_dim == 0:
            self.bucket_capacity = bucket_capacity
        else:
            self.bucket_capacity = ((bucket_capacity + digest_load_dim - 1) // digest_load_dim) * digest_load_dim
        # self.bucket_capacity = _next_power_of_2(bucket_capacity)
        if self.bucket_capacity != bucket_capacity:
            warnings.warn(f"Bucket capacity is rounded from {bucket_capacity} to {self.bucket_capacity}.", UserWarning)
        self.num_buckets = (capacity + self.bucket_capacity -1) // self.bucket_capacity
        self.capacity = self.num_buckets * self.bucket_capacity
        if self.capacity != capacity:
            warnings.warn(f"Table capacity is rounded from {capacity} to {self.capacity}.", UserWarning)

        fields_range = [0]
        fields_range.extend([x * self.capacity for x in fields_byte])
        fields_range = list(accumulate(fields_range))

        self.table_storage = torch.empty(fields_range[-1], dtype=torch.uint8, device=self.device)
        
        fields_ = table_partition(self.storage, self.bucket_capacity, self.num_buckets, fileds_type)
        # fields_ = tensor_partition(self.table_storage, fields_range, fileds_type)
        self.keys, self.values, self.scores, self.digests = fields_
    
        # self.keys = torch.empty(self.capacity, dtype=key_type, device=self.device)
        # self.values = torch.empty(self.capacity, dtype=value_type, device=self.device)
        # self.scores = torch.empty(self.capacity, dtype=torch.uint64, device=self.device)
        # self.digests = torch.empty(self.capacity, dtype=torch.uint8, device=self.device)

        self.bucket_sizes = torch.zeros(self.num_buckets, dtype=torch.int32, device=self.device)
        
        self._init_table(value_init_fn)

    
    def _init_table(
        self,
        value_init_fn
    ):
        # init keys
        empty_key = 0xFFFFFFFFFFFFFFFF
        if self.key_type == torch.int64:
            empty_key = uint64_to_int64(empty_key)
        self.keys.fill_(empty_key)

        # init values
        # not to initialize table's value by default.
        # initiliaze when want to reuse the value.
        if value_init_fn is not None:
            value_init_fn(self.values)
    
        # init scores
        empty_score = 0
        self.scores.fill_(empty_score)
    
        # init digest
        empty_digest = (murmur3_hash_64bits(empty_key) >> 32) & 0xFF
        self.digests.fill_(empty_digest)
        
    
    def lookup(
        self,
        input_keys,
        input_scores=None,
        allow_score_accum=False,
    ):
        batch = input_keys.numel()
        output_values = torch.empty(batch, dtype=self.value_type, device=input_keys.device)
        founds = torch.empty(batch, dtype=torch.bool, device=input_keys.device)
        table_lookup(
            self.table_storage, self.bucket_capacity,
            input_keys, input_scores, output_values, founds, allow_score_accum
        )
        return output_values, founds
    
    def insert(
        self,
        input_keys,
        values, # input or output
        input_scores,
        allow_value_reuse=True,
        allow_score_accum=False,
    ):
        """
        input keys have to be unique.
        """
        batch = input_keys.numel()
    
        if allow_value_reuse:
            assert values is not None, "Must provide values and will output the reused to it."
        else:
            assert values is not None, "Must provide new values when not allow to resue value."
        
        table_insert(
            self.table_storage, self.bucket_capacity, self.bucket_sizes,
            input_keys, input_scores, values, allow_value_reuse, allow_score_accum,
        )

        return None
        

    def insert_and_evict(
        self,
        input_keys,
        values, # input or output
        input_scores,
        allow_value_reuse=True,
        allow_score_accum=False,
    ):
        """
        input keys have to be unique.
        """
        batch = input_keys.numel()

        num_evicted = torch.empty(1, dtype=torch.int64, device=input_keys.device)
        evicted_keys = torch.empty(batch, dtype=self.key_type, device=input_keys.device)
        evicted_scores = torch.empty(batch, dtype=torch.uint64, device=input_keys.device)

        if allow_value_reuse:
            assert values is not None, "Must provide values and will output the reused to it."
            evicted_values = None
        else:
            assert values is not None, "Must provide new values when not allow to resue value."
            evicted_values = torch.empty(batch, dtype=self.value_type, device=input_keys.device)

        table_insert_and_evict(
            self.table_storage, self.bucket_capacity, self.bucket_sizes,
            input_keys, input_scores, values, allow_value_reuse, allow_score_accum,
            num_evicted, evicted_keys, evicted_scores, evicted_values
        )
        h_num_evicted = num_evicted.cpu().item()
        
        return h_num_evicted, evicted_keys[:h_num_evicted], evicted_values[:h_num_evicted] if evicted_values is not None else None, evicted_scores[:h_num_evicted]

    def size(self) -> int:
        return self.bucket_sizes.sum()

    def load_factor(self) -> float:
        return self.bucket_sizes.sum() / self.capacity
        

    def reserve(
        self,
        target_capacity,
    ):
        """
        Table's growth is controlled outside.
        """
        pass
    
    def memory_usage(self, mem_type=MemoryType.DEVICE) -> int:
        result = self.capacity * (self.keys.element_size() + self.values.element_size() + self.scores.element_size() + self.digests.element_size())
        result += self.num_buckets * self.bucket_sizes.element_size()
        return result
        