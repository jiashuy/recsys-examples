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

"""Read the scores out of sharded dynamicemb tables and check them as LFU counts."""

from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from dynamicemb.dump_load import find_sharded_modules, get_dynamic_emb_module


def export_table_scores(
    model: nn.Module,
    table_names: Optional[Dict[str, List[str]]] = None,
    optim: Optional[bool] = False,
    pg: Optional[dist.ProcessGroup] = None,
) -> Dict[str, Dict[int, int]]:
    """
    Load scores from dynamic embedding tables directly (without disk I/O).

    Returns:
        Dict[table_name, Dict[key, score]]: Scores organized by table name
    """

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    batch_size = 65536

    all_table_scores = {}

    for _, collection_name, sharded_module in find_sharded_modules(model, ""):
        dynamic_emb_modules = get_dynamic_emb_module(sharded_module)

        for dynamic_emb_module in dynamic_emb_modules:
            dynamic_emb_module.flush()
            storage = dynamic_emb_module.tables

            for table_idx, table_name in enumerate(dynamic_emb_module.table_names):
                if table_names is not None and table_name not in set(
                    table_names[collection_name]
                ):
                    continue

                table_key_scores = {}

                for keys, _, _, scores in storage.export_keys_values(
                    device, batch_size, table_id=table_idx
                ):
                    # For the compound (TIMESTAMP, LFU) score strategy each key
                    # exports two score words [timestamp, frequency]; the frequency
                    # is the reduction (last) word. Plain LFU exports a single
                    # frequency word.
                    if scores.dim() == 2:
                        scores = scores[:, -1]
                    for key, score in zip(keys, scores):
                        table_key_scores[int(key)] = int(score)

                all_table_scores[table_name] = table_key_scores

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    dist.barrier(group=pg, device_ids=[torch.cuda.current_device()])

    return all_table_scores


def validate_lfu_scores(
    expected_frequencies: Dict[str, Dict[int, int]],
    actual_scores: Dict[str, Dict[int, int]],
    tolerance: float = 0.0,
):
    """
    Validate that actual scores match expected frequencies for LFU strategy.

    Returns:
        (is_valid, error_message)
    """
    for table_name in expected_frequencies:
        if table_name not in actual_scores:
            assert (
                table_name in actual_scores
            ), f"Table {table_name} missing from actual scores"

        expected = expected_frequencies[table_name]
        actual = actual_scores[table_name]

        for key in actual.keys():
            exp_freq = expected[key]
            act_score = actual[key]
            assert (
                exp_freq == act_score
            ), f"Table {table_name}, Key {key}: Expected frequency: {exp_freq}, Actual score: {act_score}"
