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

"""Where a checkpoint's files are, and how to find them.
"""

import bisect
import fnmatch
import os
import re
from functools import partial
from typing import List, Optional, Sequence, Tuple


def encode_meta_json_file_path(root_path: str, table_name: str) -> str:
    return os.path.join(root_path, f"{table_name}_opt_args.json")


def encode_checkpoint_file_path(
    root_path: str, table_name: str, rank: int, world_size: int, item: str
) -> str:
    assert item in ["keys", "values", "scores", "opt_values"]
    return os.path.join(
        root_path, f"{table_name}_emb_{item}.rank_{rank}.world_size_{world_size}"
    )


def encode_counter_checkpoint_file_path(
    root_path: str, table_name: str, rank: int, world_size: int, item: str
) -> str:
    assert item in ["keys", "frequencies"]
    return os.path.join(
        root_path, f"{table_name}_counter_{item}.rank_{rank}.world_size_{world_size}"
    )


def find_files(
    root_path: str, table_name: str, suffix: str, sorted_file_names: Sequence[str]
) -> Tuple[List[str], int]:
    """Find one table's checkpoint files of *suffix* among *sorted_file_names*.

    *sorted_file_names* is the sorted listing of *root_path*, taken once by the
    caller: a checkpoint directory holds every table's files, so globbing it per
    table and per suffix lists it 6N times for N tables, which is slow where a
    listing is expensive, such as object-storage mounts. Being sorted, the names
    that can match sit together, so a lookup does not scan the whole listing.
    """
    suffix_to_encode_file_path_func = {
        "emb_keys": partial(encode_checkpoint_file_path, item="keys"),
        "emb_values": partial(encode_checkpoint_file_path, item="values"),
        "emb_scores": partial(encode_checkpoint_file_path, item="scores"),
        "opt_values": partial(encode_checkpoint_file_path, item="opt_values"),
        "counter_keys": partial(encode_counter_checkpoint_file_path, item="keys"),
        "counter_frequencies": partial(
            encode_counter_checkpoint_file_path, item="frequencies"
        ),
    }
    if suffix not in suffix_to_encode_file_path_func:
        raise RuntimeError(f"Invalid suffix: {suffix}")
    encode_file_path_func = suffix_to_encode_file_path_func[suffix]

    # v2 version
    pattern = os.path.basename(encode_file_path_func(root_path, table_name, "*", "*"))
    prefix = re.split(r"[*?[]", pattern, maxsplit=1)[0]
    start = end = bisect.bisect_left(sorted_file_names, prefix)
    while end < len(sorted_file_names) and sorted_file_names[end].startswith(prefix):
        end += 1
    files = [
        os.path.join(root_path, name)
        for name in fnmatch.filter(sorted_file_names[start:end], pattern)
    ]
    if len(files) == 0:
        return [], 0
    files = sorted(files)
    world_size = int(files[0].split(".")[-1].split("_")[-1])
    if len(files) != world_size:
        raise RuntimeError(
            f"Checkpoints is corrupted. Found {len(files)} under path {root_path} for table {table_name}, but the number of checkpointed world size is {world_size}."
        )

    for i in range(world_size):
        expected_file_path = encode_file_path_func(root_path, table_name, i, world_size)
        if expected_file_path not in set(files):
            raise RuntimeError(
                f"Checkpoints is corrupted. Expected file path {expected_file_path} for table {table_name}, but it is not found."
            )

    return files, len(files)


def get_loading_files(
    root_path: str,
    name: str,
    rank: int,
    world_size: int,
    sorted_file_names: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], List[str], List[str], int, int]:
    """Resolve one table's checkpoint files under *root_path*.

    Pass *sorted_file_names*, the sorted listing of *root_path*, when resolving
    many tables, so the directory is listed once rather than once per table.
    """
    if not os.path.exists(root_path):
        raise RuntimeError(f"can't find path to load, path:", root_path)
    if sorted_file_names is None:
        sorted_file_names = sorted(os.listdir(root_path))

    key_files, num_key_files = find_files(
        root_path, name, "emb_keys", sorted_file_names
    )
    value_files, num_value_files = find_files(
        root_path, name, "emb_values", sorted_file_names
    )
    score_files, num_score_files = find_files(
        root_path, name, "emb_scores", sorted_file_names
    )
    opt_files, num_opt_files = find_files(
        root_path, name, "opt_values", sorted_file_names
    )

    if num_key_files != num_value_files:
        assert (
            num_key_files > 0
        ), "No key files found under path {root_path} for table {name}"
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of value files."
        )

    counter_key_files, num_counter_key_files = find_files(
        root_path, name, "counter_keys", sorted_file_names
    )
    counter_freq_files, num_counter_freq_files = find_files(
        root_path, name, "counter_frequencies", sorted_file_names
    )

    if num_counter_key_files != num_counter_freq_files:
        raise RuntimeError(
            f"The number of key files of admission counter under path {root_path} for table {name} does not match the number of frequency files({num_counter_key_files}/{num_counter_freq_files})."
        )

    if num_counter_key_files > 0 and num_counter_key_files != num_key_files:
        raise RuntimeError(
            f"The number of key files under path {root_path} for table {name} does not match the number of keys files of admission counter({num_key_files}/{num_counter_key_files})."
        )

    if world_size == num_key_files:
        return (
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "keys")],
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "values")],
            [encode_checkpoint_file_path(root_path, name, rank, world_size, "scores")]
            if num_score_files == num_key_files
            else [],
            [
                encode_checkpoint_file_path(
                    root_path, name, rank, world_size, "opt_values"
                )
            ]
            if num_opt_files == num_key_files
            else [],
            [
                encode_counter_checkpoint_file_path(
                    root_path, name, rank, world_size, "keys"
                )
            ]
            if num_counter_key_files == num_key_files
            else [],
            [
                encode_counter_checkpoint_file_path(
                    root_path, name, rank, world_size, "frequencies"
                )
            ]
            if num_counter_freq_files == num_key_files
            else [],
        )
    # TODO: support skipping files.
    return (
        key_files,
        value_files,
        score_files,
        opt_files,
        counter_key_files,
        counter_freq_files,
    )
