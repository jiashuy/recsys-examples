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

"""Table-row-wise sharded DynamicEmb tables: planning, forward, backward, checkpoints.

Level: integration.   Needs: 2 GPUs (most cases) or 4 GPUs (the four-rank ones,
    skipped on smaller machines).
Covers: a table-row-wise table and a row-wise control in one
    ``EmbeddingBagCollection`` through the planner and
    ``DistributedModelParallel``: the plan (sharding type, shard counts, node
    placement, rows per rank), forward parity against row-wise, bags of
    differing length, MEAN against SUM pooling, one backward/optimizer step,
    and a checkpoint written under one sharding and loaded under the other.
Does not cover (see unit/planner/): the planner's cost model on its own.
The work is done by ``workers/table_row_wise.py`` under torchrun; its module
docstring describes each mode.

A "node" is ``LOCAL_WORLD_SIZE`` ranks, which TorchRec's
``intra_and_cross_node_pg`` and the planner's Topology both read. Two nodes of
one rank each is a faithful table-row-wise topology on a two-GPU box -- and
local_size < world_size is the only shape where table-row-wise differs from
row-wise at all. The equal case (one node of two) is run too, as the boundary
where an off-by-one between the two would show.
"""

import shutil
from pathlib import Path

import pytest
from common.cli_args import as_flags
from common.launch import run_worker

WORKER = Path(__file__).parent / "workers" / "table_row_wise.py"

ROUNDROBIN, HASH_ROUNDROBIN = "roundrobin", "hash_roundrobin"
DIST_TYPES = (ROUNDROBIN, HASH_ROUNDROBIN)


def _case(gpus: int, node_size: int, mode: str, dist_type: str, **extra):
    """One worker launch: ``gpus`` ranks in nodes of ``node_size``."""
    flags = as_flags(
        {"node_size": node_size, "mode": mode, "dist_type": dist_type, **extra}
    )
    return pytest.param(
        gpus,
        flags,
        id=f"{gpus}gpu-nodes-of-{node_size}-{mode}-{dist_type}",
    )


# --- forward / backward (CI group fwd_bwd) -------------------------------------

FORWARD_CASES = [
    # `plan` reads no key, so it is run once per topology rather than once per
    # dist_type: check_plan asserts sharding types, shard counts, which node the
    # table landed on and how many rows a rank got, and none of those is what the
    # bucketizer decides.
    #
    # Two nodes of one rank, then one node of two. The second is the degenerate
    # case -- local_size == world_size, where table-row-wise and row-wise should
    # agree exactly, and where an off-by-one between them shows up.
    _case(2, 1, "plan", HASH_ROUNDROBIN),
    _case(2, 2, "plan", HASH_ROUNDROBIN),
    # `parity` and `jagged` do read keys, so both rules are run. Bags of
    # differing length, empty ones included, reach the bucketizer's edges, and
    # the two rules bucket by different arithmetic.
    *[
        _case(gpus, node, mode, dist_type)
        for dist_type in DIST_TYPES
        for gpus, node, mode in ((2, 1, "parity"), (2, 1, "jagged"), (2, 2, "parity"))
    ],
    # MEAN against SUM, the one absolute check here. Pooling happens after the
    # lookup -- the kernel's downgrade to SUM and the EBC's own mean callback --
    # so like `plan` it does not depend on where a key was sent, and a rule that
    # sent one to the wrong rank would already have failed `parity` above.
    _case(2, 1, "pooling", HASH_ROUNDROBIN),
    # The backward: the gradient divisions depend on the node size and the node
    # count, not on the rule that placed the key, so one rule and both
    # topologies. The four-rank one below is the only place the two divisions
    # are both > 1.
    _case(2, 1, "grad", HASH_ROUNDROBIN),
    # Four ranks split two ways, so a node holds more than one rank. The only
    # topology here where intra-node is itself a group.
    *[
        _case(4, 2, mode, HASH_ROUNDROBIN)
        for mode in ("plan", "parity", "jagged", "grad")
    ],
]


@pytest.mark.parametrize("gpus,flags", FORWARD_CASES)
def test_table_row_wise_forward_backward(gpus, flags):
    """Planner output, forward parity vs row-wise, pooling and gradient step."""
    run_worker(WORKER, gpus, flags)


# --- checkpoints (CI group load_dump) ------------------------------------------

# Checkpoints, including across geometries: a checkpoint written by a
# table-row-wise table has to load into a row-wise one and back, which is the
# resharding path M4 added and the one section 8.1's defect lived on. These are
# the three combinations that involve a table-row-wise table, written out rather
# than looped over: row_wise to row_wise is the path every other checkpoint
# suite already covers, and both halves of it are reached by the two crossings.
SHARDING_PAIRS = (
    ("table_row_wise", "table_row_wise"),
    ("table_row_wise", "row_wise"),
    ("row_wise", "table_row_wise"),
)

CHECKPOINT_CASES = [
    pytest.param(
        dist_type,
        dump_sharding,
        load_sharding,
        id=f"{dist_type}-{dump_sharding}-to-{load_sharding}",
    )
    for dist_type in DIST_TYPES
    for dump_sharding, load_sharding in SHARDING_PAIRS
]


@pytest.mark.load_dump
@pytest.mark.parametrize("dist_type,dump_sharding,load_sharding", CHECKPOINT_CASES)
def test_table_row_wise_checkpoint_round_trip(
    tmp_path, dist_type, dump_sharding, load_sharding
):
    """Dump under one sharding, load under another (2 ranks, nodes of 1)."""
    save_path = tmp_path / "ckpt"
    shutil.rmtree(save_path, ignore_errors=True)
    for mode, sharding in (("dump", dump_sharding), ("load", load_sharding)):
        flags = as_flags(
            {
                "node_size": 1,
                "mode": mode,
                "trw_sharding": sharding,
                "dist_type": dist_type,
                "save_path": save_path,
            }
        )
        run_worker(WORKER, 2, flags)
