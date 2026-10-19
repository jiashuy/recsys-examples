# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""What the DynamicEmb tables cost each rank, and what TorchRec is left with.

Single process, no GPU and no process group: these are the arithmetic half of
planning, and the point of having them as functions over a plan is that they can
be checked without one.
"""

import copy
from dataclasses import fields
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
import torchrec
from dynamicemb.dynamicemb_config import (
    DynamicEmbTableOptions,
    get_group_value_bytes_by_tier,
    get_local_value_bytes_by_tier,
)
from dynamicemb.planner import plan
from dynamicemb.planner.plan import (
    _STORAGE_HAS_SSD,
    make_storage,
    module_without_tables,
    optimizer_types,
    per_rank_storage,
    shard_rank,
    table_fanout,
    table_ranks,
    topology_minus,
)
from dynamicemb.planner.planners import (
    DynamicEmbeddingShardingPlanner,
    DynamicEmbParameterConstraints,
    DynamicEmbParameterSharding,
)
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torch.distributed.optim import (
    _apply_optimizer_in_backward as apply_optimizer_in_backward,
)
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner.types import PlannerError, Topology
from torchrec.distributed.types import (
    EnumerableShardingSpec,
    ParameterSharding,
    ShardingType,
    ShardMetadata,
)
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig

WORLD_SIZE = 4
LOCAL_SIZE = 2
DIM = 16
ROWS = 1024
GB = 1024**3
# What one rank of the machine under test has, as `_topology` builds it.
HBM_PER_RANK = 80 * GB
DDR_PER_RANK = 128 * GB


def _options(rows: int = ROWS, hbm_budget: int = 0) -> DynamicEmbTableOptions:
    return DynamicEmbTableOptions(
        max_capacity=rows,
        dim=DIM,
        embedding_dtype=torch.float32,
        local_hbm_for_values=hbm_budget,
    )


class _Planned(NamedTuple):
    """A table as the planner keeps it: decisions on the plan record, and the
    options it was planned with beside them rather than on it."""

    sharding: ParameterSharding
    options: DynamicEmbTableOptions


def _sharding(ranks, rows: int = ROWS, hbm_budget: int = 0) -> _Planned:
    """One shard of `rows` rows on each of `ranks`, and its options."""
    return _Planned(
        ParameterSharding(
            sharding_type=ShardingType.ROW_WISE.value,
            compute_kernel="customized_kernel",
            ranks=list(ranks),
            sharding_spec=EnumerableShardingSpec(
                [
                    ShardMetadata(
                        shard_sizes=[rows, DIM],
                        shard_offsets=[rows * i, 0],
                        placement=f"rank:{rank}/cuda:{rank % LOCAL_SIZE}",
                    )
                    for i, rank in enumerate(ranks)
                ]
            ),
        ),
        _options(rows, hbm_budget),
    )


def _spent(planned, optimizer_types, world_size: int = WORLD_SIZE, poolings=None):
    """`per_rank_storage` over a {name: _Planned} mapping.

    The plan and the options go in as two arguments because that is how the
    planner holds them -- the plan record carries decisions, and the options
    live with the constraints they came from.
    """
    return per_rank_storage(
        {name: p.sharding for name, p in planned.items()},
        {name: p.options for name, p in planned.items()},
        optimizer_types=optimizer_types,
        world_size=world_size,
        poolings=poolings,
    )


def _topology(hbm: int = HBM_PER_RANK, ddr: int = DDR_PER_RANK) -> Topology:
    return Topology(
        world_size=WORLD_SIZE,
        local_world_size=LOCAL_SIZE,
        compute_device="cuda",
        hbm_cap=hbm,
        ddr_cap=ddr,
    )


def test_shard_rank_reads_both_placement_forms():
    """Placement is a _remote_device or the string it was built from.

    ``ShardMetadata.__post_init__`` parses a string argument, so the normal
    construction gives the first form. The string form is set afterwards to
    cover the case TorchRec itself still guards for
    (``distributed/embeddingbag.py:184-185``).
    """
    shard = ShardMetadata(
        shard_offsets=[0, 0], shard_sizes=[ROWS, DIM], placement="rank:3/cuda:1"
    )
    # __post_init__ parsed the string into a _remote_device.
    assert not isinstance(shard.placement, str)
    assert shard_rank(shard) == 3

    # A later assignment does not go through __post_init__, so it stays a str --
    # the form TorchRec itself still guards for.
    shard.placement = "rank:3/cuda:1"
    assert isinstance(shard.placement, str)
    assert shard_rank(shard) == 3


class TestPerRankStorage:
    """What each rank spends on the DynamicEmb tables planned onto it."""

    def test_row_wise_table_charges_every_rank_equally(self):
        per_rank = _spent(
            {"t": _sharding(range(WORLD_SIZE))}, optimizer_types={"t": None}
        )
        assert len(per_rank) == WORLD_SIZE
        values = ROWS * DIM * 4  # float32, no optimizer state
        assert all(s == make_storage(hbm=0, ddr=values, ssd=0) for s in per_rank)

    def test_a_table_on_one_node_charges_only_that_node(self):
        """The reason this is per rank and not one repeated figure.

        Under TABLE_ROW_WISE a table lives on one node's ranks. Charging every rank
        for it would reserve four nodes' worth of room for one node's table.
        """
        per_rank = _spent(
            {"t": _sharding([2, 3])},  # the second node only
            optimizer_types={"t": None},
        )
        values = ROWS * DIM * 4
        assert per_rank[0] == make_storage()
        assert per_rank[1] == make_storage()
        assert per_rank[2] == make_storage(hbm=0, ddr=values, ssd=0)
        assert per_rank[3] == make_storage(hbm=0, ddr=values, ssd=0)

    def test_optimizer_state_lands_entirely_in_ddr(self):
        """local_hbm_for_values is a fixed budget, so the optimizer's extra bytes
        cannot come out of HBM -- they all spill to the host tier."""
        hbm_budget = ROWS * DIM * 4  # exactly the values, nothing spare
        without = _spent(
            {"t": _sharding(range(WORLD_SIZE), hbm_budget=hbm_budget)},
            optimizer_types={"t": None},
        )
        with_adam = _spent(
            {"t": _sharding(range(WORLD_SIZE), hbm_budget=hbm_budget)},
            optimizer_types={"t": OptimType.ADAM},
        )

        assert without[0].ddr == 0
        assert with_adam[0].hbm == without[0].hbm == hbm_budget
        assert with_adam[0].ddr > 0

    def test_a_shard_outside_the_world_is_an_error(self):
        with pytest.raises(ValueError, match="outside the world"):
            _spent({"t": _sharding([WORLD_SIZE])}, optimizer_types={"t": None})


class TestTopologyMinus:
    """Taking what the DynamicEmb tables cost out of the machine.

    Per rank, into a copy, and refusing rather than clamping -- TorchRec's own
    reservation only ever looks at ``devices[0]``.
    """

    def test_topology_minus_subtracts_per_rank_and_copies(self):
        topology = _topology()
        per_rank = [
            make_storage(hbm=i * GB, ddr=2 * i * GB, ssd=0) for i in range(WORLD_SIZE)
        ]
        before = copy.deepcopy([d.storage for d in topology.devices])

        reduced = topology_minus(topology, per_rank)

        for i, device in enumerate(reduced.devices):
            assert device.storage.hbm == HBM_PER_RANK - i * GB
            assert device.storage.ddr == DDR_PER_RANK - 2 * i * GB
        # the caller's topology is an input they may reuse
        assert [d.storage for d in topology.devices] == before

    def test_topology_minus_refuses_an_overdrawn_rank(self):
        """Neither clamped nor passed on. Clamping reads as "no room, plan
        accordingly" and the plan comes back looking feasible; passing it on relies
        on TorchRec's reservation, which tests `devices[0]` and no other rank."""
        topology = _topology(hbm=1 * GB)
        with pytest.raises(PlannerError, match="do not fit"):
            topology_minus(topology, [make_storage(hbm=4 * GB)] * WORLD_SIZE)

    def test_an_overdraft_away_from_rank_zero_is_still_caught(self):
        """The case TorchRec's own check misses: rank 0 is fine, a later rank is
        not. A table-row-wise table pinned to any node but the first lands here."""
        spent = [make_storage(hbm=0)] * (WORLD_SIZE - 1) + [make_storage(hbm=200 * GB)]
        with pytest.raises(PlannerError, match=f"Rank {WORLD_SIZE - 1} has"):
            topology_minus(_topology(), spent)

    def test_an_exact_fit_is_not_an_overdraft(self):
        topology = _topology(hbm=4 * GB, ddr=8 * GB)
        reduced = topology_minus(
            topology, [make_storage(hbm=4 * GB, ddr=8 * GB)] * WORLD_SIZE
        )
        assert reduced.devices[0].storage.hbm == 0
        assert reduced.devices[0].storage.ddr == 0

    def test_negative_tiers_reads_ssd_where_the_field_exists(self, monkeypatch):
        """The ssd half of the guard, on a TorchRec whose Storage has no ssd.

        `storage_ssd` reads the flag at call time, so forcing it exercises the
        branch that the container this usually runs in cannot reach. Without this
        the ssd case would be skipped everywhere and never execute at all."""
        monkeypatch.setattr(plan, "_STORAGE_HAS_SSD", True)

        assert plan._negative_tiers(SimpleNamespace(hbm=1, ddr=1, ssd=-1)) == ["ssd"]
        assert plan._negative_tiers(SimpleNamespace(hbm=-1, ddr=1, ssd=-1)) == [
            "hbm",
            "ssd",
        ]
        assert plan._negative_tiers(SimpleNamespace(hbm=1, ddr=1, ssd=0)) == []

    @pytest.mark.skipif(
        not _STORAGE_HAS_SSD, reason="Storage.ssd exists from TorchRec 1.5.0"
    )
    def test_an_ssd_overdraft_is_caught(self):
        """`Storage.__sub__` subtracts all three tiers. A guard reading only hbm
        and ddr would hand the planner a negative ssd without a word.

        `ssd_cap=0` is stated rather than left out: TorchRec defaults it to 2 TB
        (`planner/constants.py:27`), which no plausible per-rank cost overdraws."""
        topology = Topology(
            world_size=WORLD_SIZE,
            local_world_size=LOCAL_SIZE,
            compute_device="cuda",
            hbm_cap=HBM_PER_RANK,
            ddr_cap=DDR_PER_RANK,
            ssd_cap=0,
        )
        assert topology.devices[0].storage.ssd == 0, "nothing to draw on"

        with pytest.raises(PlannerError, match="overdraws ssd"):
            topology_minus(topology, [make_storage(ssd=1 * GB)] * WORLD_SIZE)

    def test_topology_minus_rejects_a_length_mismatch(self):
        with pytest.raises(ValueError, match="entries but the topology has"):
            topology_minus(_topology(), [make_storage()] * (WORLD_SIZE - 1))


class TestTheBroadcastSeam:
    """Decisions travel, options do not.

    Rank 0 broadcasts a plan of plain ``ParameterSharding``s; every rank then
    completes its own entries from the options it kept, because
    ``external_storage`` is a live handle and ``score_function`` a callable.
    """

    def test_the_plan_record_carries_decisions_and_nothing_else(self):
        """`_insert_dynamicemb_plan` puts its own record straight into the plan, so
        the record has to be exactly what the plan carries. Options on it would
        ride the broadcast, where `external_storage` is a live handle this process
        opened and `score_function` is a Python callable."""
        planned = _sharding(range(WORLD_SIZE))

        assert type(planned.sharding) is ParameterSharding
        assert not hasattr(planned.sharding, "dynamicemb_options")

    def test_the_options_are_added_on_arrival(self):
        """The other half: a rank completes the decisions it was sent with the
        options it kept. Every inherited field has to survive, which is why
        `with_options` copies them through `fields(ParameterSharding)` rather than
        naming them."""
        planned = _sharding(range(WORLD_SIZE))

        completed = DynamicEmbParameterSharding.with_options(
            planned.sharding, planned.options
        )

        assert isinstance(completed, DynamicEmbParameterSharding)
        assert completed.dynamicemb_options is planned.options
        for f in fields(ParameterSharding):
            if f.name == "sharding_spec":
                continue  # copied, not shared -- the next test is about that
            assert getattr(completed, f.name) == getattr(
                planned.sharding, f.name
            ), f.name

        # What TorchRec will ask it for, once the kernel is built.
        assert completed.get_additional_fused_params() == {
            "dynamicemb_options": planned.options
        }

    def test_the_completed_entry_does_not_share_metadata_with_the_record(self):
        """The one field the loop above cannot check, because `==` cannot tell a
        copy from the original.

        DMP rewrites placements in place when its device is `meta`
        (`embeddingbag.py:183-188`), and on rank 0 the plan still holds the
        planner's own record -- a broadcast hands the leader back its own object
        rather than a round trip (`collective_utils.py:65-67`). Sharing the spec
        would let the first DMP rewrite what a second `plan` call hands out, on
        rank 0 and nowhere else.
        """
        planned = _sharding(range(WORLD_SIZE))

        completed = DynamicEmbParameterSharding.with_options(
            planned.sharding, planned.options
        )

        # Two steps: `is not` alone passes for a shallow copy, which still shares
        # the `shards` list and the ShardMetadata objects DMP actually writes to.
        assert completed.sharding_spec is not planned.sharding.sharding_spec
        completed.sharding_spec.shards[0].placement = "rank:7/cuda:7"
        assert (
            str(planned.sharding.sharding_spec.shards[0].placement) != "rank:7/cuda:7"
        ), "a write through the completed entry reached the planner's own record"


# --- where the refusal happens ---------------------------------------------


def _dynamicemb_planner(topology: Topology, table: str, hbm_budget: int):
    """A planner whose only table is a DynamicEmb one of the given HBM budget."""
    return DynamicEmbeddingShardingPlanner(
        topology=topology,
        constraints={
            table: DynamicEmbParameterConstraints(
                sharding_types=[ShardingType.ROW_WISE.value],
                use_dynamicemb=True,
                dynamicemb_options=DynamicEmbTableOptions(
                    global_hbm_for_values=hbm_budget,
                ),
            )
        },
    )


class TestWhereTheRefusalHappens:
    """Overcommitment is refused in step 1, which every rank runs.

    Raised inside the broadcast it would unwind rank 0 alone and leave the
    others waiting for a plan that is never sent, so the one check that
    catches it would surface as a hang.
    """

    def test_an_overdraft_is_refused_before_the_broadcast(self):
        """`_plan_torchrec` runs on rank 0 alone, inside the broadcast in
        `collective_plan`. A refusal raised in there unwinds rank 0 and leaves
        every other rank waiting on a broadcast that will never be sent, so the
        one check that catches overcommitment would surface as a hang.

        `_plan_dynamicemb` runs on every rank, so that is where it belongs. This
        asserts it by calling step 1 alone -- `_plan_torchrec` is never reached."""
        # Row-wise over WORLD_SIZE ranks: 2**26 rows of DIM fp32 each, so 4 GiB a
        # rank, against a budget that allows it and a rank that has 1 GB.
        module = _ebc(["big"], rows=2**28)
        sharders = [EmbeddingBagCollectionSharder()]
        planner = _dynamicemb_planner(_topology(hbm=1 * GB), "big", 64 * GB)

        with pytest.raises(PlannerError, match="do not fit"):
            planner._plan_dynamicemb(module, sharders)

    def test_a_plan_that_fits_leaves_the_caller_topology_whole(self):
        """Step 1 deducts into a Topology of its own. `self._topology` still says
        what the machine has, because a second `plan` call has to start there."""
        topology = _topology()
        module = _ebc(["small"])
        planner = _dynamicemb_planner(topology, "small", 1 * GB)

        planner._plan_dynamicemb(module, [EmbeddingBagCollectionSharder()])

        assert planner._reduced_topology is not None
        assert planner._topology.devices[0].storage.hbm == HBM_PER_RANK
        assert planner._reduced_topology.devices[0].storage.hbm < HBM_PER_RANK


# --- module_without_tables -------------------------------------------------


def _ebc(names, device=torch.device("meta"), rows: int = ROWS):
    return torchrec.EmbeddingBagCollection(
        tables=[
            EmbeddingBagConfig(
                name=name,
                embedding_dim=DIM,
                num_embeddings=rows,
                feature_names=[f"f_{name}"],
            )
            for name in names
        ],
        device=device,
    )


class _Model(torch.nn.Module):
    def __init__(self, ebc):
        super().__init__()
        self.sparse = torch.nn.Module()
        self.sparse.ebc = ebc
        self.dense = torch.nn.Linear(DIM, DIM, device=torch.device("meta"))


def _sharders():
    return [EmbeddingBagCollectionSharder()]


class TestModuleWithoutTables:
    """The copy TorchRec is given to plan: the DynamicEmb tables gone, every
    path the plan is keyed by still where it was."""

    def test_named_tables_are_gone_from_the_copy(self):
        model = _Model(_ebc(["a", "b", "c"]))
        reduced = module_without_tables(model, {"a", "c"}, _sharders())

        assert set(reduced.sparse.ebc.embedding_bags.keys()) == {"b"}
        # the caller's module is untouched
        assert set(model.sparse.ebc.embedding_bags.keys()) == {"a", "b", "c"}

    def test_the_tree_keeps_its_shape_so_plan_paths_still_match(self):
        model = _Model(_ebc(["a", "b"]))
        reduced = module_without_tables(model, {"a"}, _sharders())

        assert [n for n, _ in reduced.named_children()] == [
            n for n, _ in model.named_children()
        ]
        assert type(reduced.sparse.ebc) is type(model.sparse.ebc)
        # only the path to the pruned collection is copied; the rest is shared
        assert reduced.dense is model.dense

    def test_a_collection_of_only_dynamicemb_tables_becomes_empty(self):
        """It stays in the tree. Removing it would change the paths the plan is
        keyed by, and an empty collection simply yields no shardable parameters."""
        model = _Model(_ebc(["a", "b"]))
        reduced = module_without_tables(model, {"a", "b"}, _sharders())

        assert len(reduced.sparse.ebc.embedding_bags) == 0
        assert _sharders()[0].shardable_parameters(reduced.sparse.ebc) == {}

    def test_nothing_is_copied_when_nothing_is_removed(self):
        model = _Model(_ebc(["a"]))
        assert module_without_tables(model, set(), _sharders()) is model
        assert module_without_tables(model, {"absent"}, _sharders()) is model


# --- table_ranks: which ranks hold a table, and what its rows divide by ----

# A second, roomier machine, because table_ranks is about nodes and the one
# above has only two ranks in each. Named apart from WORLD_SIZE/LOCAL_SIZE
# rather than one underscore away from them.
WORLD_8, LOCAL_4 = 8, 4  # two nodes of four
RW = ShardingType.ROW_WISE.value
TRW = ShardingType.TABLE_ROW_WISE.value


class TestTableRanks:
    """Which ranks hold a table, and how many ways its rows are split."""

    def test_row_wise_is_every_rank(self):
        assert table_ranks(RW, None, WORLD_8, LOCAL_4) == list(range(WORLD_8))

    def test_table_row_wise_is_one_node(self):
        assert table_ranks(TRW, 0, WORLD_8, LOCAL_4) == [0, 1, 2, 3]
        assert table_ranks(TRW, 1, WORLD_8, LOCAL_4) == [4, 5, 6, 7]

    @pytest.mark.parametrize(
        "sharding_type, node_index, world, local, message",
        [
            # Not ignored: a row-wise table is on every rank, so naming a node for
            # it means the caller expected something the plan will not do.
            # `_plan_dynamicemb` leans on this -- it writes the placer's answer back
            # into the options for table-row-wise tables only, because clearing a
            # row-wise table's `node_index` to None first would turn the refusal
            # into a silent correction.
            (RW, 0, WORLD_8, LOCAL_4, "on every rank of every node"),
            # The mirror: one node has to be named, and there is no default worth
            # guessing.
            (TRW, None, WORLD_8, LOCAL_4, "needs a node_index"),
            (TRW, 2, WORLD_8, LOCAL_4, "outside the 2 nodes"),
            # A world that is not a whole number of nodes has no node to place on.
            (TRW, 0, 7, LOCAL_4, "does not divide into nodes"),
            # DynamicEmb plans these two sharding types and no others.
            (
                ShardingType.COLUMN_WISE.value,
                None,
                WORLD_8,
                LOCAL_4,
                "not 'column_wise'",
            ),
        ],
        ids=[
            "node_index_on_row_wise",
            "table_row_wise_without_node_index",
            "node_index_past_the_last_node",
            "world_is_not_whole_nodes",
            "unsupported_sharding_type",
        ],
    )
    def test_table_ranks_refuses(
        self, sharding_type, node_index, world, local, message
    ):
        with pytest.raises(ValueError, match=message):
            table_ranks(sharding_type, node_index, world, local)

    def test_a_table_row_wise_table_only_costs_its_own_node(self):
        """§4.1 and §4.2 meeting §2: the placement decides who pays, and
        per_rank_storage reads it off the shard metadata."""
        ranks = table_ranks(TRW, 1, WORLD_8, LOCAL_4)
        spent = _spent(
            {"t": _sharding(ranks)}, optimizer_types={"t": None}, world_size=WORLD_8
        )

        # No HBM budget and no optimizer state, so the whole shard is host bytes.
        a_shard = make_storage(hbm=0, ddr=ROWS * DIM * 4, ssd=0)
        for rank in range(WORLD_8):
            expected = a_shard if rank in ranks else make_storage()
            assert spent[rank] == expected, f"rank {rank}"

    def test_a_row_wise_table_is_split_the_same_way_it_always_was(self):
        """`table_fanout` replaced a bare `dist.get_world_size()`. For row-wise it
        has to be that same number, or every existing table resizes.

        The node size is varied to say that row-wise does not read it at all."""
        for world, local in [(8, 4), (2, 2), (1, 1), (16, 8), (4, 1)]:
            assert table_fanout(RW, world, local) == world

    def test_a_table_row_wise_table_is_split_over_one_node(self):
        """The mirror: a node's worth of ranks, whichever node it turns out to be.

        Not depending on *which* node is what lets a table's capacity and HBM
        budget be settled before the placer has decided where to put it."""
        for world, local in [(8, 4), (2, 2), (1, 1), (16, 8), (4, 1)]:
            assert table_fanout(TRW, world, local) == local

    @pytest.mark.parametrize(
        "sharding_type, world, local, message",
        [
            # DynamicEmb plans row-wise and table-row-wise and no others.
            (ShardingType.COLUMN_WISE.value, WORLD_8, LOCAL_4, "not 'column_wise'"),
            # A world that is not a whole number of nodes has no node to split
            # over -- and neither does a node of no ranks.
            (TRW, 7, LOCAL_4, "does not divide into nodes"),
            (TRW, WORLD_8, 0, "does not divide into nodes"),
        ],
        ids=[
            "unsupported_sharding_type",
            "world_is_not_whole_nodes",
            "node_of_no_ranks",
        ],
    )
    def test_table_fanout_refuses(self, sharding_type, world, local, message):
        with pytest.raises(ValueError, match=message):
            table_fanout(sharding_type, world, local)


class TestGroupedTiers:
    """What a set of tables that share a TBE costs a rank.

    The runtime sums the group's sizes and the group's HBM budgets and decides
    once, so a table with budget to spare lends it. Costing each table alone
    gives ``sum(min(total, budget))``, at most ``min(sum total, sum budget)``.
    """

    def test_a_spare_budget_is_lent_to_the_group(self):
        """The runtime sums both sides and decides once, so a table whose budget
        exceeds its size lends the remainder. Costing each alone under-states HBM,
        and that is the direction that lets through a plan the machine will not
        hold."""
        rows_bytes = ROWS * DIM * 4
        small = _options(hbm_budget=2 * rows_bytes)  # needs 1x, allowed 2x
        large = _options(rows=4 * ROWS, hbm_budget=0)  # needs 4x, allowed none

        alone = [get_group_value_bytes_by_tier([(o, None)]) for o in (small, large)]
        per_table_hbm = sum(h for h, _ in alone)
        per_table_ddr = sum(d for _, d in alone)

        grouped_hbm, grouped_ddr = get_group_value_bytes_by_tier(
            [(small, None), (large, None)]
        )

        assert per_table_hbm < grouped_hbm, "per-table must under-state HBM here"
        assert per_table_ddr > grouped_ddr
        # The group spends every byte of budget it has, on the total it has.
        assert grouped_hbm == 2 * rows_bytes
        assert grouped_ddr == 5 * rows_bytes - 2 * rows_bytes

    def test_a_group_of_one_is_the_single_table_answer(self):
        o = _options(hbm_budget=ROWS * DIM * 4 // 2)
        assert get_group_value_bytes_by_tier(
            [(o, None)]
        ) == get_local_value_bytes_by_tier(o, None)

    def test_an_unsized_table_reports_its_budget(self):
        """No dim or capacity means nothing to size from, so it fills its own
        budget rather than counting as free."""
        bare = DynamicEmbTableOptions(local_hbm_for_values=7)
        assert get_group_value_bytes_by_tier([(bare, None)]) == (7, 0)

    def test_per_rank_storage_groups_what_shares_a_rank(self):
        """Two tables with the same grouping key on the same ranks are one group,
        so the spare budget of one covers the other."""
        rows_bytes = ROWS * DIM * 4
        small = _options(hbm_budget=2 * rows_bytes)
        large = _options(rows=4 * ROWS, hbm_budget=0)

        def sharding(options):
            ranks = list(range(WORLD_SIZE))
            return ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel="customized_kernel",
                ranks=ranks,
                sharding_spec=EnumerableShardingSpec(
                    [
                        ShardMetadata(
                            shard_sizes=[options.max_capacity, DIM],
                            shard_offsets=[options.max_capacity * i, 0],
                            placement=f"rank:{r}/cuda:{r % LOCAL_SIZE}",
                        )
                        for i, r in enumerate(ranks)
                    ]
                ),
            )

        spent = per_rank_storage(
            {"small": sharding(small), "large": sharding(large)},
            {"small": small, "large": large},
            optimizer_types={"small": None, "large": None},
            world_size=WORLD_SIZE,
        )
        grouped = get_group_value_bytes_by_tier([(small, None), (large, None)])
        assert all(s == make_storage(hbm=grouped[0], ddr=grouped[1]) for s in spent)


# --- optimizer_types: which optimizer each table trains with ----------------


def _ec(names, device=torch.device("meta")):
    return torchrec.EmbeddingCollection(
        tables=[
            EmbeddingConfig(
                name=name,
                embedding_dim=DIM,
                num_embeddings=ROWS,
                feature_names=[f"f_{name}"],
            )
            for name in names
        ],
        device=device,
    )


class _MixedModel(torch.nn.Module):
    """One pooled collection and one sequence collection, as a real model has.

    On CPU rather than meta because `apply_optimizer_in_backward` registers
    hooks on the parameters.
    """

    def __init__(self):
        super().__init__()
        self.pooled = _ebc(["p0", "p1"], device=torch.device("cpu"))
        self.sequence = _ec(["s0"], device=torch.device("cpu"))


def _mixed_sharders(pooled_optimizer=None, sequence_optimizer=None):
    return [
        EmbeddingBagCollectionSharder(
            fused_params=None
            if pooled_optimizer is None
            else {"optimizer": pooled_optimizer}
        ),
        EmbeddingCollectionSharder(
            fused_params=None
            if sequence_optimizer is None
            else {"optimizer": sequence_optimizer}
        ),
    ]


class TestOptimizerTypes:
    """Where a table's optimizer comes from, when the model is sharded by more
    than one sharder and they need not agree."""

    @pytest.fixture
    def model(self):
        return _MixedModel()

    def test_each_collection_reads_its_own_sharders_optimizer(self, model):
        """A table follows the sharder that claims it.

        Flattening the list to a single value -- taking the last sharder that
        names an optimizer -- would hand every table one module's answer, and
        the one it picked would depend on the order the caller happened to pass
        them in.
        """
        sharders = _mixed_sharders(
            pooled_optimizer=OptimType.ADAM,
            sequence_optimizer=OptimType.EXACT_ROWWISE_ADAGRAD,
        )

        assert optimizer_types(model, sharders) == {
            "p0": OptimType.ADAM,
            "p1": OptimType.ADAM,
            "s0": OptimType.EXACT_ROWWISE_ADAGRAD,
        }

        # and the answer does not depend on the order they were passed in
        assert optimizer_types(model, list(reversed(sharders))) == optimizer_types(
            model, sharders
        )

    def test_a_parameter_optimizer_wins_over_its_sharder(self, model):
        apply_optimizer_in_backward(
            torch.optim.SGD, model.pooled.parameters(), {"lr": 0.1}
        )

        assert optimizer_types(
            model, _mixed_sharders(pooled_optimizer=OptimType.ADAM)
        ) == {
            "p0": OptimType.EXACT_SGD,
            "p1": OptimType.EXACT_SGD,
            "s0": None,
        }

    def test_an_optimizer_torchrec_cannot_name_falls_back_to_the_sharder(self, model):
        """DynamicEmb's own FTRL is one of these: applied to the parameter, but
        with no EmbOptimType to convert to. The sharder's answer is better than
        none."""
        apply_optimizer_in_backward(
            torch.optim.RMSprop, model.pooled.parameters(), {"lr": 0.1}
        )

        assert optimizer_types(
            model, _mixed_sharders(pooled_optimizer=OptimType.ADAM)
        ) == {
            "p0": OptimType.ADAM,
            "p1": OptimType.ADAM,
            "s0": None,
        }

    def test_a_table_no_sharder_claims_is_not_reported(self, model):
        """`shardable_parameters` is the same accessor `module_without_tables`
        prunes by, so what is planned and what is costed are the same set."""
        assert set(optimizer_types(model, [EmbeddingBagCollectionSharder()])) == {
            "p0",
            "p1",
        }
