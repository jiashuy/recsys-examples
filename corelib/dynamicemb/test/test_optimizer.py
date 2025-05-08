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

import random
from typing import Any, Dict, List, Tuple

import pytest
import torch
from dynamicemb import DynamicEmbTableOptions, OptimizerArgs
from dynamicemb.dynamicemb_config import *
from dynamicemb.optimizer import (
    AdaGradDynamicEmbeddingOptimizer,
    AdamDynamicEmbeddingOptimizer,
    BaseDynamicEmbeddingOptimizer,
    OptimizerArgs,
    RowWiseAdaGradDynamicEmbeddingOptimizer,
    SGDDynamicEmbeddingOptimizer,
)
from dynamicemb_extensions import (
    DynamicEmbTable,
    find,
    find_or_insert,
    insert_or_assign,
)


class TorchSGDDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
        scores: Optional[List[int]] = None,
    ) -> None:
        for ht in hashtables:
            if ht not in self._hashtables:
                raise ValueError(
                    f"DynamicEmb ERROR:Hashtable {ht} not found in hashtables in class {self.__class__.__name__}."
                )

        lr = self._opt_args.learning_rate
        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]

            grad = grads[i]
            indice = indices[i]
            num_indice = indice.shape[0]
            grad_shape = grads[i].shape

            weight_tensor = torch.zeros(
                grad_shape, dtype=table_option.embedding_dtype, device=grads[i].device
            )
            find_or_insert(ht, num_indice, indice, weight_tensor)
            weight_tensor -= lr * grad
            insert_or_assign(ht, num_indice, indice, weight_tensor)

    def get_opt_args(self):
        ret_args = {"lr": self._opt_args.learning_rate}
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        return


class TorchAdamDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

        self._iterations: int = 0

        for table_option in self._table_options:
            table_option.initializer_args = DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT, value=0.0
            )

        self._state_dict["m"] = []
        self._state_dict["v"] = []
        self._create_tables(self._state_dict["m"])
        self._create_tables(self._state_dict["v"])

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
        scores: Optional[List[int]] = None,
    ) -> None:
        for ht in hashtables:
            if ht not in self._table_state_map.keys():
                raise ValueError(
                    f"DynamicEmb ERROR:Hashtable {ht} not found in _table_state_map in class {self.__class__.__name__}."
                )

        self._iterations += 1
        lr = self._opt_args.learning_rate
        beta1 = self._opt_args.beta1
        beta2 = self._opt_args.beta2
        weight_decay = self._opt_args.weight_decay
        eps = self._opt_args.eps

        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]
            m_ht = self._state_dict["m"][state_idx]
            v_ht = self._state_dict["v"][state_idx]

            indice = indices[i]
            grad = grads[i]
            num_indice = indice.shape[0]
            grad_shape = grads[i].shape

            m_tensor = torch.zeros(
                grad_shape, dtype=table_option.embedding_dtype, device=grads[i].device
            )
            v_tensor = torch.zeros(
                grad_shape, dtype=table_option.embedding_dtype, device=grads[i].device
            )
            weight_tensor = torch.zeros(
                grad_shape, dtype=table_option.embedding_dtype, device=grads[i].device
            )

            find_or_insert(m_ht, num_indice, indice, m_tensor)
            find_or_insert(v_ht, num_indice, indice, v_tensor)
            find_or_insert(ht, num_indice, indice, weight_tensor)

            m_tensor = beta1 * m_tensor + (1 - beta1) * grad
            v_tensor = beta2 * v_tensor + (1 - beta2) * grad.pow(2)
            m_hat = m_tensor / (1 - beta1**self._iterations)
            v_hat = v_tensor / (1 - beta2**self._iterations)
            weight_update = lr * (
                m_hat / (torch.sqrt(v_hat) + eps) + weight_decay * weight_tensor
            )
            weight_tensor -= weight_update

            insert_or_assign(ht, num_indice, indice, weight_tensor)
            insert_or_assign(m_ht, num_indice, indice, m_tensor)
            insert_or_assign(v_ht, num_indice, indice, v_tensor)

    def get_opt_args(self):
        ret_args = {
            "lr": self._opt_args.learning_rate,
            "iters": self._iterations,
            "beta1": self._opt_args.beta1,
            "beta2": self._opt_args.beta2,
            "eps": self._opt_args.eps,
            "weight_decay": self._opt_args.weight_decay,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._iterations = get_required_arg(args, "iters")
        self._opt_args.beta1 = get_required_arg(args, "beta1")
        self._opt_args.beta2 = get_required_arg(args, "beta2")
        self._opt_args.eps = get_required_arg(args, "eps")
        self._opt_args.weight_decay = get_required_arg(args, "weight_decay")
        return


class TorchAdagradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

        for table_option in self._table_options:
            table_option.initializer_args = DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT, value=0.0
            )
        self._state_dict["Gt"] = []
        self._create_tables(self._state_dict["Gt"])

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
        scores: Optional[List[int]] = None,
    ) -> None:
        for ht in hashtables:
            if ht not in self._table_state_map.keys():
                raise ValueError(
                    f"DynamicEmb ERROR: Hashtable {ht} not found in _table_state_map in class {self.__class__.__name__}."
                )

        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]
            gt_ht = self._state_dict["Gt"][state_idx]

            indice = indices[i]
            grad = grads[i]
            num_indice = indice.shape[0]
            grad_shape = grad.shape

            gt_tensor = torch.zeros(
                grad_shape, dtype=table_option.embedding_dtype, device=grad.device
            )
            weight_tensor = torch.zeros(
                grad_shape, dtype=table_option.embedding_dtype, device=grad.device
            )

            find_or_insert(gt_ht, num_indice, indice, gt_tensor)
            find_or_insert(ht, num_indice, indice, weight_tensor)

            gt_tensor.add_(grad.pow(2))  # v_tensor += grad^2

            update = lr * grad / (gt_tensor.sqrt().add_(eps))

            weight_tensor.sub_(update)  # weight -= update

            insert_or_assign(ht, num_indice, indice, weight_tensor)
            insert_or_assign(gt_ht, num_indice, indice, gt_tensor)

    def get_opt_args(self):
        ret_args = {
            "lr": self._opt_args.learning_rate,
            "eps": self._opt_args.eps,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._opt_args.eps = get_required_arg(args, "eps")

        return


class TorchRowWiseAdagradDynamicEmbeddingOptimizer(BaseDynamicEmbeddingOptimizer):
    def __init__(
        self,
        opt_args: OptimizerArgs,
        table_options: List[DynamicEmbTableOptions],
        hashtables: List[DynamicEmbTable],
    ) -> None:
        super().__init__(opt_args, table_options, hashtables)

        for table_option in self._table_options:
            old_dim = table_option.dim
            old_global_hbm_for_values = table_option.global_hbm_for_values
            old_local_hbm_for_values = table_option.local_hbm_for_values

            new_global_hbm_for_values = old_global_hbm_for_values // old_dim
            new_local_hbm_for_values = old_local_hbm_for_values // old_dim

            table_option.initializer_args = DynamicEmbInitializerArgs(
                mode=DynamicEmbInitializerMode.CONSTANT,
                value=0.0,
            )
            table_option.dim = 1
            table_option.global_hbm_for_values = new_global_hbm_for_values
            table_option.local_hbm_for_values = new_local_hbm_for_values

        self._state_dict["Gt"] = []
        self._create_tables(self._state_dict["Gt"])

    def update(
        self,
        hashtables: List[DynamicEmbTable],
        indices: List[torch.Tensor],
        grads: List[torch.Tensor],
        scores: Optional[List[int]] = None,
    ) -> None:
        for ht in hashtables:
            if ht not in self._table_state_map.keys():
                raise ValueError(
                    f"DynamicEmb ERROR: Hashtable {ht} not found in _table_state_map in class {self.__class__.__name__}."
                )

        lr = self._opt_args.learning_rate
        eps = self._opt_args.eps

        for i, ht in enumerate(hashtables):
            state_idx = self._table_state_map[ht]
            table_option = self._table_options[state_idx]
            gt_ht = self._state_dict["Gt"][state_idx]

            indice = indices[i]
            grad = grads[i]
            num_indice = indice.shape[0]
            grad_shape = grad.shape
            D = grad_shape[-1]

            gt_tensor = torch.zeros(
                (num_indice, 1), dtype=table_option.embedding_dtype, device=grad.device
            )
            weight_tensor = torch.zeros(
                grad_shape, dtype=table_option.embedding_dtype, device=grad.device
            )

            find_or_insert(gt_ht, num_indice, indice, gt_tensor)
            find_or_insert(ht, num_indice, indice, weight_tensor)

            grad_sq_sum = grad.pow(2).sum(dim=1, keepdim=True)
            grad_sq_avg = grad_sq_sum / D
            gt_tensor.add_(grad_sq_avg)

            lr_scaled = lr / (gt_tensor.sqrt() + eps)
            update = lr_scaled * grad

            weight_tensor.sub_(update)  # weight -= update

            insert_or_assign(ht, num_indice, indice, weight_tensor)
            insert_or_assign(gt_ht, num_indice, indice, gt_tensor)

    def get_opt_args(self):
        ret_args = {
            "lr": self._opt_args.learning_rate,
            "eps": self._opt_args.eps,
        }
        return ret_args

    def set_opt_args(self, args: Dict[str, Any]):
        self._opt_args.learning_rate = get_required_arg(args, "lr")
        self._opt_args.eps = get_required_arg(args, "eps")

        return


def generate_random_data(
    num_tables: int, length: int, embedding_dim: List[int], index_range: Tuple[int, int]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    indices = []
    grads = []
    insert_values = []
    for i in range(num_tables):
        indice = torch.tensor(
            random.sample(range(index_range[0], index_range[1]), length),
            dtype=torch.int64,
            device="cuda",
        )
        grad = torch.randn((length, embedding_dim[i]), device="cuda")
        insert_value = torch.randn((length, embedding_dim[i]), device="cuda")
        indices.append(indice)
        grads.append(grad)
        insert_values.append(insert_value)
    return indices, grads, insert_values


# Helper function to initialize hash tables
def initialize_hashtables(
    num_tables: int, table_options: List[DynamicEmbTableOptions]
) -> List[DynamicEmbTable]:
    hashtables = []
    for i in range(num_tables):
        table_option = table_options[i]
        hashtables.append(create_dynamicemb_table(table_option))
    return hashtables


# Helper function to compare tensors
def compare_tensors(t1: torch.Tensor, t2: torch.Tensor, rtol=1e-05, atol=1e-06):
    return torch.allclose(t1, t2, rtol=rtol, atol=atol)


# Function to test optimizers
@pytest.mark.parametrize(
    "optimizer_class",
    [
        (SGDDynamicEmbeddingOptimizer, TorchSGDDynamicEmbeddingOptimizer),
        (AdamDynamicEmbeddingOptimizer, TorchAdamDynamicEmbeddingOptimizer),
        (AdaGradDynamicEmbeddingOptimizer, TorchAdagradDynamicEmbeddingOptimizer),
        (
            RowWiseAdaGradDynamicEmbeddingOptimizer,
            TorchRowWiseAdagradDynamicEmbeddingOptimizer,
        ),
    ],
)
@pytest.mark.parametrize("num_tables", [4])
@pytest.mark.parametrize("length", [1024])
@pytest.mark.parametrize(
    "embedding_dim", [[16, 128, 256, 512], [17, 128, 256, 513], [17, 33, 155, 511]]
)
@pytest.mark.parametrize("num_tests", [10])
@pytest.mark.parametrize("index_min", [0])
@pytest.mark.parametrize("index_max", [4096])
def test_optimizer(
    optimizer_class, num_tables, length, embedding_dim, num_tests, index_min, index_max
):
    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")
    # Define optimizer arguments
    opt_args = OptimizerArgs(
        learning_rate=0.01, eps=1e-8, beta1=0.9, beta2=0.999, weight_decay=0.1
    )

    table_options = [
        DynamicEmbTableOptions(
            index_type=torch.int64,
            embedding_dtype=torch.float32,
            evict_strategy=DynamicEmbEvictStrategy.LRU,
            dim=embedding_dim[i],
            init_capacity=int(1 * 1024 * 1024),
            max_capacity=int(1 * 1024 * 1024),
            local_hbm_for_values=4 * 1024**3,
            bucket_capacity=1024,
            device_id=-1,
        )
        for i in range(num_tables)
    ]

    # print(f"optimizer_class = {optimizer_class}")
    # print(f"opt_args = {opt_args}")
    # print(f"table_options = {table_options}")
    # print(f"num_tables = {num_tables}")
    # print(f"length = {length}")
    # print(f"embedding_dim = {embedding_dim}")
    # print(f"num_tests = {num_tests}")
    # print(f"index_min = {index_min}")
    # print(f"index_max = {index_max}")
    # print(f"device = {device}")

    dynamicemb_optimizer_class = optimizer_class[0]
    torch_optimizer_class = optimizer_class[1]
    # Initialize hash tables
    hashtables_for_torch = initialize_hashtables(num_tables, table_options)
    hashtables_for_dynamicemb = initialize_hashtables(num_tables, table_options)

    opt_for_torch = torch_optimizer_class(opt_args, table_options, hashtables_for_torch)
    opt_for_dynamicemb = dynamicemb_optimizer_class(
        opt_args, table_options, hashtables_for_dynamicemb
    )

    for test_num in range(num_tests):
        print(f"Running test {test_num + 1}/{num_tests}")

        # Generate random indices, gradients, and insert values
        indices, grads, insert_values = generate_random_data(
            num_tables, length, embedding_dim, index_range=(index_min, index_max)
        )

        # Insert initial values into hash tables
        for i, ht in enumerate(hashtables_for_torch):
            length = indices[i].shape[0]
            find_or_insert(ht, length, indices[i], insert_values[i])

        for i, ht in enumerate(hashtables_for_dynamicemb):
            length = indices[i].shape[0]
            find_or_insert(ht, length, indices[i], insert_values[i])

        # Create found_tensor and result_tensor lists
        found_tensors_for_torch = [
            torch.zeros_like(tensor, dtype=torch.bool, device=device)
            for tensor in indices
        ]
        result_tensors_for_torch = [
            torch.zeros_like(tensor, device=device) for tensor in insert_values
        ]

        found_tensors_for_dynamicemb = [
            torch.zeros_like(tensor, dtype=torch.bool, device=device)
            for tensor in indices
        ]
        result_tensors_for_dynamicemb = [
            torch.zeros_like(tensor, device=device) for tensor in insert_values
        ]

        for i, ht in enumerate(hashtables_for_torch):
            length = indices[i].shape[0]
            insert_or_assign(ht, length, indices[i], insert_values[i])

        for i, ht in enumerate(hashtables_for_dynamicemb):
            length = indices[i].shape[0]
            insert_or_assign(ht, length, indices[i], insert_values[i])

        # check insert result is right
        for i, ht in enumerate(hashtables_for_torch):
            length = indices[i].shape[0]
            find(
                ht,
                length,
                indices[i],
                result_tensors_for_torch[i],
                found_tensors_for_torch[i],
            )

        for i, ht in enumerate(hashtables_for_dynamicemb):
            length = indices[i].shape[0]
            find(
                ht,
                length,
                indices[i],
                result_tensors_for_dynamicemb[i],
                found_tensors_for_dynamicemb[i],
            )

        # Synchronize CUDA
        torch.cuda.synchronize()

        # Validation
        # Check if all found tensors are True
        for tensor in found_tensors_for_torch:
            assert tensor.all(), "Not all values in found_tensors_for_torch are True"
        for tensor in found_tensors_for_dynamicemb:
            assert (
                tensor.all()
            ), "Not all values in found_tensors_for_dynamicemb are True"

        # Check if result_tensors_for_torch and result_tensors_for_dynamicemb match insert_values
        for i, tensor in enumerate(result_tensors_for_torch):
            assert compare_tensors(
                tensor, insert_values[i]
            ), f"result_tensors_for_torch[{i}] does not match insert_values[{i}]"

        for i, tensor in enumerate(result_tensors_for_dynamicemb):
            assert compare_tensors(
                tensor, insert_values[i]
            ), f"result_tensors_for_dynamicemb[{i}] does not match insert_values[{i}]"

        # Check if result_tensors_for_torch and result_tensors_for_dynamicemb are equal
        for i in range(len(result_tensors_for_torch)):
            assert compare_tensors(
                result_tensors_for_torch[i], result_tensors_for_dynamicemb[i]
            ), f"result_tensors_for_torch[{i}] does not match result_tensors_for_dynamicemb[{i}]"

        print("Assign and Find success!")

        opt_for_torch.update(hashtables_for_torch, indices, grads)
        opt_for_dynamicemb.update(hashtables_for_dynamicemb, indices, grads)
        torch.cuda.synchronize()

        found_weights_for_torch = [
            torch.zeros_like(tensor, dtype=torch.bool, device=device)
            for tensor in indices
        ]
        result_weights_for_torch = [
            torch.zeros_like(tensor, device=device) for tensor in insert_values
        ]

        found_weights_for_dynamicemb = [
            torch.zeros_like(tensor, dtype=torch.bool, device=device)
            for tensor in indices
        ]
        result_weights_for_dynamicemb = [
            torch.zeros_like(tensor, device=device) for tensor in insert_values
        ]

        # check insert result is right
        for i, ht in enumerate(hashtables_for_torch):
            length = indices[i].shape[0]
            find(
                ht,
                length,
                indices[i],
                result_weights_for_torch[i],
                found_weights_for_torch[i],
            )

        for i, ht in enumerate(hashtables_for_dynamicemb):
            length = indices[i].shape[0]
            find(
                ht,
                length,
                indices[i],
                result_weights_for_dynamicemb[i],
                found_weights_for_dynamicemb[i],
            )

        # Synchronize CUDA
        torch.cuda.synchronize()

        # Validation
        # Check if all found tensors are True
        for tensor in found_weights_for_torch:
            assert tensor.all(), "Not all values in found_tensors_for_torch are True"
        for tensor in found_weights_for_dynamicemb:
            assert (
                tensor.all()
            ), "Not all values in found_tensors_for_dynamicemb are True"

        # Check if result_tensors_for_torch and result_tensors_for_dynamicemb are equal
        for i in range(len(result_weights_for_torch)):
            if not compare_tensors(
                result_weights_for_torch[i], result_weights_for_dynamicemb[i]
            ):
                print(
                    f"Difference found at index {i} result_weights_for_torch = {result_weights_for_torch[i]} result_weights_for_dynamicemb = {result_weights_for_dynamicemb[i]}"
                )
                diff = result_weights_for_torch[i] - result_weights_for_dynamicemb[i]
                print(f"Difference tensor: {diff}")
                raise AssertionError(
                    f"result_weights_for_torch[{i}] does not match result_weights_for_dynamicemb[{i}] within tolerance"
                )
            else:
                print(
                    f"result_weights_for_torch[{i}] matches result_weights_for_dynamicemb[{i}] within tolerance"
                )

        print("All tensors match successfully.")

        state_names = opt_for_torch.state_names()
        if len(state_names) > 0:
            for state_name in state_names:
                tmp_state_hts_for_torch = opt_for_torch.get_state_by_name(state_name)
                tmp_state_hts_for_dynamicemb = opt_for_dynamicemb.get_state_by_name(
                    state_name
                )
                found_states_for_torch = [
                    torch.zeros_like(tensor, dtype=torch.bool, device=device)
                    for tensor in indices
                ]
                result_states_for_torch = [
                    torch.zeros_like(tensor, device=device) for tensor in insert_values
                ]

                found_states_for_dynamicemb = [
                    torch.zeros_like(tensor, dtype=torch.bool, device=device)
                    for tensor in indices
                ]
                result_states_for_dynamicemb = [
                    torch.zeros_like(tensor, device=device) for tensor in insert_values
                ]

                # check insert result is right
                for i, ht in enumerate(tmp_state_hts_for_torch):
                    length = indices[i].shape[0]
                    find(
                        ht,
                        length,
                        indices[i],
                        result_states_for_torch[i],
                        found_states_for_torch[i],
                    )

                for i, ht in enumerate(tmp_state_hts_for_dynamicemb):
                    length = indices[i].shape[0]
                    find(
                        ht,
                        length,
                        indices[i],
                        result_states_for_dynamicemb[i],
                        found_states_for_dynamicemb[i],
                    )

                # Synchronize CUDA
                torch.cuda.synchronize()

                # Validation
                # Check if all found tensors are True
                for tensor in found_states_for_torch:
                    assert (
                        tensor.all()
                    ), "Not all values in found_tensors_for_torch are True"
                for tensor in found_states_for_dynamicemb:
                    assert (
                        tensor.all()
                    ), "Not all values in found_tensors_for_dynamicemb are True"

                for i in range(len(result_states_for_torch)):
                    if not compare_tensors(
                        result_states_for_torch[i], result_states_for_dynamicemb[i]
                    ):
                        print(f"Difference found at index {i}")
                        diff = (
                            result_states_for_torch[i] - result_states_for_dynamicemb[i]
                        )
                        print(f"Difference tensor: {diff}")
                        raise AssertionError(
                            f"result_states_for_torch[{i}] does not match result_states_for_dynamicemb[{i}] within tolerance"
                        )
                    else:
                        print(
                            f"result_states_for_torch[{i}] matches result_states_for_dynamicemb[{i}] within tolerance"
                        )
