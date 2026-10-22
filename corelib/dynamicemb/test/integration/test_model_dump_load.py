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

"""Dump a sharded dynamicemb model and load the checkpoint back, end to end.

Level: integration.   Needs: 1 GPU, 2 GPUs, or 8 GPUs (per case; the 8-GPU
    cases are skipped on smaller machines).
Covers: ``DynamicEmbDump`` on one world size and ``DynamicEmbLoad`` on the same
    or another (1 <-> 8 ranks), for every optimizer, with and without optimizer
    state and admission counters, for every score strategy. The loaded model
    must hold the dumped keys with their scores (LFU counts, or monotonic
    timestamp/step order), the admission counters must match, the dist_type must
    reach the sharding and input-dist, and the forward output (and a further
    training step when optimizer state was dumped) must equal the reference
    model's. Also hash_roundrobin dumped and reloaded at the same world size.
Does not cover (see unit/table/test_table_dump_load.py): dump/load of a bare
    table, without the model stack; (see test_table_row_wise.py): checkpoints
    of table-row-wise tables.
The work is done by ``workers/model_dump_load.py`` under torchrun, once with
``--mode dump`` and once with ``--mode load``.
"""

from pathlib import Path

import pytest
from common.cli_args import as_flags
from common.launch import require_gpus, run_worker

pytestmark = pytest.mark.load_dump

WORKER = Path(__file__).parent / "workers" / "model_dump_load.py"

NUM_GPUS = (1, 8)
OPTIMIZERS = ("adam", "sgd", "rowwise_adagrad")
INCLUDE_OPTIM = (True, False)
INCLUDE_COUNTER = (True, False)
SCORE_STRATEGIES = ("timestamp", "lfu", "step")

_MODEL = {
    "num_embedding_collections": 4,
    "num_embeddings": ",".join(["1000000"] * 6),
    "multi_hot_sizes": ",".join(["10"] * 6),
    "embedding_dim": 16,
}

# Dumped once per (world size, config) and shared by the loads that read it, so
# the 1 -> 1 and 1 -> 8 cases do not dump the same model twice. Keyed by the
# checkpoint path; a case whose dump failed leaves nothing in here.
_DUMPED = set()


def _dump_then_load(path, dump_gpus, load_gpus, options):
    """Dump at ``dump_gpus`` ranks (unless done), load at ``load_gpus`` ranks."""
    require_gpus(max(dump_gpus, load_gpus))
    if path not in _DUMPED:
        run_worker(WORKER, dump_gpus, as_flags({**options, "mode": "dump"}))
        _DUMPED.add(path)
    run_worker(WORKER, load_gpus, as_flags({**options, "mode": "load"}))


def _ckpt_options(save_path, optimizer, include_optim, include_counter, score):
    return {
        **_MODEL,
        "optimizer_type": optimizer,
        "score_strategy": score,
        "optim": str(include_optim),
        "counter": str(include_counter),
        "save_path": save_path,
    }


# Every optimizer x optimizer-state x counter x score strategy, dumped on
# `dump_gpus` ranks and loaded on `load_gpus`: 1 -> 1 and 8 -> 8 are same-size
# reloads, 1 -> 8 and 8 -> 1 reshard the checkpoint.
RESHARD_CASES = [
    pytest.param(
        dump_gpus,
        load_gpus,
        optimizer,
        include_optim,
        include_counter,
        score,
        id=(
            f"dump{dump_gpus}-load{load_gpus}-{optimizer}"
            f"-{'optim' if include_optim else 'no-optim'}"
            f"-{'counter' if include_counter else 'no-counter'}-{score}"
        ),
    )
    for dump_gpus in NUM_GPUS
    for load_gpus in NUM_GPUS
    for optimizer in OPTIMIZERS
    for include_optim in INCLUDE_OPTIM
    for include_counter in INCLUDE_COUNTER
    for score in SCORE_STRATEGIES
]


@pytest.mark.parametrize(
    "dump_gpus,load_gpus,optimizer,include_optim,include_counter,score",
    RESHARD_CASES,
)
def test_dump_load_across_world_sizes(
    tmp_path_factory,
    dump_gpus,
    load_gpus,
    optimizer,
    include_optim,
    include_counter,
    score,
):
    """Load reproduces keys, scores, counters and forward output, same or other size."""
    path = str(
        tmp_path_factory.getbasetemp()
        / f"ckpt_{optimizer}_{dump_gpus}_{include_optim}_{include_counter}_{score}"
    )
    options = _ckpt_options(path, optimizer, include_optim, include_counter, score)
    _dump_then_load(path, dump_gpus, load_gpus, options)


# hash_roundrobin, dumped and reloaded at the same world size. Multi-rank is
# the point: a rank's checkpoint file holds the keys with murmur(k) % ws == r,
# so a loader applying the roundrobin rule instead keeps only the ~1/ws of them
# that also satisfy k % ws == r and silently drops the rest. Single rank cannot
# see that -- there is no filtering to get wrong.
@pytest.mark.parametrize("num_gpus", (1, 2), ids=lambda n: f"{n}gpu")
def test_hash_roundrobin_same_world_size(tmp_path, num_gpus):
    """A hash_roundrobin checkpoint reloads without silently dropping keys."""
    options = {
        "num_embedding_collections": 1,
        "num_embeddings": "1000000",
        "multi_hot_sizes": "10",
        "embedding_dim": 16,
        "optimizer_type": "sgd",
        "score_strategy": "step",
        "dist_type": "hash_roundrobin",
        "optim": "False",
        "counter": "False",
        "save_path": str(tmp_path / "ckpt"),
    }
    require_gpus(num_gpus)
    for mode in ("dump", "load"):
        run_worker(WORKER, num_gpus, as_flags({**options, "mode": mode}))
