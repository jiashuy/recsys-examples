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

"""Forward of dynamicemb-sharded TorchRec models, checked after the all-to-all.

Level: integration.   Needs: 2 GPUs (the big-table cases 4 and 8 GPUs).
Covers: a ``DistributedModelParallel`` model with row-wise sharded dynamicemb
tables, in three shapes -- pooled (``EmbeddingBagCollection``), sequence
(``EmbeddingCollection``, with and without index dedup, ending on an empty batch
on rank 0) and two sequence collections in one model -- over embedding dim
(aligned and not), output dtype flag, number of dynamicemb tables, bag-size
profiles (including empty features), and large tables on 4 and 8 GPUs. The
workers' ``Debugger`` recomputes the expected embeddings from the input keys.
Does not cover (see unit/module/test_batched_table_backward.py): the backward /
optimizer update of the batched tables. Sharded-vs-TorchRec training equivalence
is in test_twin_module.py.
"""

from pathlib import Path

import pytest
from common.launch import run_worker

WORKER = Path(__file__).parent / "workers" / "embedding_forward.py"

# The Debugger only checks the first PYHKV_DEBUG_ITER iterations of a run.
DEBUG_ENV = {"PYHKV_DEBUG": "1", "PYHKV_DEBUG_ITER": "10"}

COMMON = ["--print_sharding_plan", "--optimizer_type", "adam"]
BIG_TABLES = [
    "--batch_size=1024",
    "--num_embeddings_per_feature=8388608,4194304,524288,1048576",
]


def case(nproc, kind, *args, id):
    return pytest.param(nproc, ["--kind", kind, *COMMON, *args], id=id)


POOLED = [
    case(2, "pooled", id="2gpu-default"),
    case(2, "pooled", "--embedding_dim", "129", id="2gpu-dim129"),
    case(2, "pooled", "--output_dtype=bf16", id="2gpu-output-bf16"),
    case(2, "pooled", "--dynamicemb_num=4", id="2gpu-4-dynamicemb-tables"),
    case(2, "pooled", "--multi_hot_sizes=20,1,101,49", id="2gpu-multi-hot-20-1-101-49"),
    case(
        2,
        "pooled",
        "--dynamicemb_num=3",
        "--data_parallel_embeddings=3",
        id="2gpu-data-parallel-table",
    ),
    case(4, "pooled", *BIG_TABLES, id="4gpu-big-tables"),
    case(8, "pooled", *BIG_TABLES, id="8gpu-big-tables"),
]

SEQUENCE = [
    case(2, "multi_sequence", "--use_index_dedup", "True", id="2gpu-two-collections"),
    case(2, "sequence", "--use_index_dedup", "True", id="2gpu-dedup"),
    case(2, "sequence", "--use_index_dedup", "False", id="2gpu-no-dedup"),
    case(
        2,
        "sequence",
        "--use_index_dedup",
        "True",
        "--embedding_dim",
        "129",
        id="2gpu-dedup-dim129",
    ),
    case(
        2,
        "sequence",
        "--use_index_dedup",
        "False",
        "--embedding_dim",
        "129",
        id="2gpu-no-dedup-dim129",
    ),
    case(
        2,
        "sequence",
        "--use_index_dedup",
        "True",
        "--output_dtype=bf16",
        id="2gpu-dedup-output-bf16",
    ),
    case(
        2,
        "sequence",
        "--use_index_dedup",
        "True",
        "--dynamicemb_num=4",
        id="2gpu-dedup-4-dynamicemb-tables",
    ),
    case(
        2,
        "sequence",
        "--use_index_dedup",
        "True",
        "--multi_hot_sizes=20,0,101,49",
        id="2gpu-dedup-empty-feature-1",
    ),
    case(
        2,
        "sequence",
        "--use_index_dedup",
        "True",
        "--multi_hot_sizes=20,49,101,0",
        id="2gpu-dedup-empty-feature-3",
    ),
    case(
        2,
        "sequence",
        "--use_index_dedup",
        "True",
        "--dynamicemb_num=3",
        "--data_parallel_embeddings=3",
        id="2gpu-dedup-data-parallel-table",
    ),
    case(4, "sequence", "--use_index_dedup", "True", *BIG_TABLES, id="4gpu-big-tables"),
    case(8, "sequence", "--use_index_dedup", "True", *BIG_TABLES, id="8gpu-big-tables"),
]


@pytest.mark.parametrize("nproc,args", POOLED)
def test_pooled_forward(nproc, args):
    """Pooled lookups (EmbeddingBagCollection) match the reference after all2all."""
    run_worker(WORKER, nproc, args, env=DEBUG_ENV)


@pytest.mark.parametrize("nproc,args", SEQUENCE)
def test_sequence_forward(nproc, args):
    """Sequence lookups (EmbeddingCollection[s]) match the reference after all2all."""
    run_worker(WORKER, nproc, args, env=DEBUG_ENV)
