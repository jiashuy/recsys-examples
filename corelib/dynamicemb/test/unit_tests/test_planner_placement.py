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

"""Which node a table-row-wise table lands on.

Single process, no GPU and no process group: the choice is arithmetic over
declared sizes, which is why it is a component of its own.
"""

import pytest
import torch
from dynamicemb.dynamicemb_config import DynamicEmbTableOptions
from dynamicemb.planner.placement import BalancedNodePlacer, TableToPlace
from dynamicemb.planner.plan import make_storage
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torchrec.distributed.planner.types import CustomTopologyData, Topology
from torchrec.modules.embedding_configs import PoolingType

GB = 1024**3
DIM = 16
LOCAL, NODES = 4, 2
WORLD = LOCAL * NODES


def _topology(hbm: int = 80 * GB, ddr: int = 128 * GB) -> Topology:
    return Topology(
        world_size=WORLD,
        local_world_size=LOCAL,
        compute_device="cuda",
        hbm_cap=hbm,
        ddr_cap=ddr,
    )


def _uneven_topology(hbm_per_rank_gb):
    """A machine whose GPUs are not all the same size.

    TorchRec takes a per-device capacity through ``custom_topology_data`` and
    carries it into ``devices[r].storage``, so this is a shape the planner can
    really be handed -- and the only way the ranks of one node differ, since
    what the placer is told has been spent is the row-wise tables' cost, which
    is the same on every rank of the world.
    """
    return Topology(
        world_size=WORLD,
        local_world_size=LOCAL,
        compute_device="cuda",
        ddr_cap=128 * GB,
        custom_topology_data=CustomTopologyData(
            data={"hbm_cap": [gb * GB for gb in hbm_per_rank_gb]},
            world_size=WORLD,
        ),
    )


def _nothing_spent():
    return [make_storage(hbm=0, ddr=0, ssd=0) for _ in range(WORLD)]


def _table(
    name,
    size_gb,
    pinned_to=None,
    budget_gb=None,
    dist_type="roundrobin",
    embedding_dtype=torch.float32,
    optimizer_type=None,
    pooling=None,
):
    """A table weighing `size_gb` per rank, as float32.

    Its HBM budget defaults to its size, so it sits entirely in HBM and
    `_table(name, N)` costs N GB of it -- which is what the tests that predate
    pooling assume. Pass `budget_gb` to make a table that spills to the host,
    or one with budget to spare for a groupmate.

    `size_gb` sizes the row count, not the bytes: with a narrower
    `embedding_dtype` the same rows weigh proportionally less. The last four
    arguments are the fields two tables must match on to fuse, and the row
    count is held fixed across them so a test can vary one and nothing else.
    """
    rows = size_gb * GB // (DIM * 4) or 1
    budget = size_gb if budget_gb is None else budget_gb
    return TableToPlace(
        name=name,
        options=DynamicEmbTableOptions(
            max_capacity=rows,
            dim=DIM,
            embedding_dtype=embedding_dtype,
            local_hbm_for_values=budget * GB,
            dist_type=dist_type,
        ),
        optimizer_type=optimizer_type,
        pooling=pooling,
        pinned_to=pinned_to,
    )


class TestPins:
    """`DynamicEmbTableOptions.node_index`, honoured and checked.

    A pin is not a decision the placer makes, so it is taken first and taken
    whole -- but it says *which* node, not that the node will hold the table,
    and everything else has to be fitted around what it took. Those are four
    separate promises and they are kept together here because a change to one
    is usually a change to another.
    """

    def test_a_pin_that_fits_is_honoured(self):
        assert BalancedNodePlacer().place(
            [_table("a", 70, pinned_to=1)], _topology(hbm=80 * GB), _nothing_spent()
        ) == {"a": 1}

    def test_the_rest_are_fitted_around_it(self):
        """`a` would have taken node 0 on size alone -- it is the bigger, and
        both nodes are empty. The pin moves it and `b` takes what is left."""
        chosen = BalancedNodePlacer().place(
            [_table("a", 30, pinned_to=1), _table("b", 20)],
            _topology(),
            _nothing_spent(),
        )
        assert chosen == {"a": 1, "b": 0}

    @pytest.mark.parametrize(
        "table, message",
        [
            # A node index this topology does not have.
            (_table("a", 1, pinned_to=5), "outside the 2 nodes"),
            # Charging a pin unchecked leaves that node's ranks negative, and
            # TorchRec's own reservation looks for that on `devices[0]` alone
            # (`storage_reservations.py:509`) -- so a pin to any node but the
            # first would go unnoticed and the plan would be returned.
            (_table("a", 200, pinned_to=1), "cannot hold it"),
        ],
        ids=["off_the_topology", "node_too_small"],
    )
    def test_a_pin_that_cannot_be_kept_is_refused(self, table, message):
        with pytest.raises(ValueError, match=message):
            BalancedNodePlacer().place([table], _topology(), _nothing_spent())

    def test_room_already_taken_by_an_earlier_pin_counts(self):
        """The fit is checked against what is left, not against an empty
        machine. Both are pinned to node 0 and together they overflow it, so
        the second is refused -- by name, since the message is otherwise the
        same for either."""
        tables = [_table("first", 60, pinned_to=0), _table("second", 40, pinned_to=0)]
        with pytest.raises(ValueError, match="'second'"):
            BalancedNodePlacer().place(tables, _topology(hbm=80 * GB), _nothing_spent())


def test_a_table_that_fits_nowhere_says_so():
    with pytest.raises(ValueError, match="No node can hold"):
        BalancedNodePlacer().place([_table("a", 200)], _topology(), _nothing_spent())


def test_largest_first_onto_the_emptiest_node():
    """Three different sizes, so this is the one that pins the sort *direction*:
    ask for the smallest first and `c` lands on node 0 instead.

    `test_equal_tables_spread_rather_than_stack` cannot see that -- its tables
    weigh the same, so sorting them is a no-op -- and
    `test_the_answer_does_not_depend_on_the_order_it_was_asked` only sees
    whether there is a sort at all.
    """
    chosen = BalancedNodePlacer().place(
        [_table("a", 30), _table("b", 20), _table("c", 10)],
        _topology(),
        _nothing_spent(),
    )
    # a is biggest and goes first; both nodes are empty so it takes the lower.
    # b then sees node 1 emptier, and c follows b only because node 1 still has
    # more room than node 0 does after a.
    assert chosen == {"a": 0, "b": 1, "c": 1}


def test_equal_tables_spread_rather_than_stack():
    """Four tables of one size, none near a rank's capacity, so nothing here is
    about fitting -- only about which node gets asked.

    A first-fit placer would put all four on node 0, which is the arrangement
    `BalancedNodePlacer` refuses on TorchRec's behalf rather than ours. Equal
    sizes also make this the only test that reaches both tie-breaks: by name
    when two tables weigh the same, by lower node index when two nodes have the
    same room. Only the balance is asserted, not which table went where -- that
    is `test_largest_first_onto_the_emptiest_node`'s to say.
    """
    chosen = BalancedNodePlacer().place(
        [_table("a", 10), _table("b", 10), _table("c", 10), _table("d", 10)],
        _topology(),
        _nothing_spent(),
    )
    assert sorted(chosen.values()) == [0, 0, 1, 1]


def test_what_is_spent_evenly_does_not_shift_the_choice():
    """`committed` carries the row-wise DynamicEmb tables, which are on every
    rank. Spent evenly it costs both nodes the same, so it leaves the ranking
    between them exactly where it was -- which is why the placer is never told
    about row-wise tables as tables."""
    evenly = [make_storage(hbm=40 * GB, ddr=0, ssd=0) for _ in range(WORLD)]
    tables = [_table("a", 10), _table("b", 10)]
    assert BalancedNodePlacer().place(
        tables, _topology(), evenly
    ) == BalancedNodePlacer().place(tables, _topology(), _nothing_spent())


def test_what_is_spent_can_still_rule_a_table_out():
    """What it does change is whether a table-row-wise table fits at all.

    The same table against the same machine, twice, differing only in
    `committed`: with the rank untouched it has room to spare, and with most of
    the rank already spent there is nowhere to put it.
    """
    per_rank, spent, table = 80, 60, 30  # GB
    assert table < per_rank, "it fits on an untouched rank"
    assert table > per_rank - spent, "and not on one already this full"

    big, topology = _table("big", table), _topology(hbm=per_rank * GB)
    everywhere = [make_storage(hbm=spent * GB, ddr=0, ssd=0) for _ in range(WORLD)]

    # Which node it lands on is `test_largest_first_onto_the_emptiest_node`'s
    # to say; all this one needs is that it lands somewhere.
    assert "big" in BalancedNodePlacer().place([big], topology, _nothing_spent())

    with pytest.raises(ValueError, match="No node can hold"):
        BalancedNodePlacer().place([big], topology, everywhere)


def test_a_node_is_charged_on_every_one_of_its_ranks():
    """A table-row-wise table charges every rank of its node, so what it leaves
    the next one is a per-rank figure rather than a share of a node total.

    Two calls rather than one because that is the situation every table after
    the first is already in: `place` loads a node through `held`, and `spare`
    subtracts that from each of its ranks. Handing the first table's cost to
    the second call as `committed` is the same arithmetic with the state
    written out, and it is what makes the second table's answer node 1.
    """
    per_rank, first, second = 80, 60, 30  # GB
    assert first <= per_rank, "the first table fits on an untouched rank"
    assert second > per_rank - first, "and leaves too little for the second"

    topology = _topology(hbm=per_rank * GB)
    assert BalancedNodePlacer().place(
        [_table("a", first)], topology, _nothing_spent()
    ) == {"a": 0}

    on_node_0 = [
        make_storage(hbm=first * GB if r < LOCAL else 0, ddr=0, ssd=0)
        for r in range(WORLD)
    ]
    assert BalancedNodePlacer().place([_table("b", second)], topology, on_node_0) == {
        "b": 1
    }


def test_a_node_is_only_as_large_as_its_smallest_gpu():
    """A table-row-wise table is split over every rank of its node, so the node
    holds it only if its *tightest* rank does. Room on the other three is room
    the table cannot reach.

    This is what `capacity`'s `min` is for, and the only arrangement that tells
    `min` from `max`: a node whose ranks differ. Reading the roomiest rank
    instead would put the table where one of its shards cannot fit, and the
    plan would look feasible until the run.
    """
    # Node 0: three large GPUs and one small one. Node 1: four middling ones.
    topology = _uneven_topology([80, 80, 80, 20] + [40] * LOCAL)

    # 30 GB a rank. Node 0 has more room in total and more room on three of its
    # four ranks, and still cannot take it.
    assert BalancedNodePlacer().place(
        [_table("t", 30)], topology, _nothing_spent()
    ) == {"t": 1}

    # 50 GB a rank fits on no node, though node 0 has three ranks that could
    # hold it.
    with pytest.raises(ValueError, match="No node can hold"):
        BalancedNodePlacer().place([_table("t", 50)], topology, _nothing_spent())


def test_the_answer_does_not_depend_on_the_order_it_was_asked():
    """It runs on every rank; two ranks that disagreed would shard one table
    onto two nodes."""
    tables = [
        _table("a", 30),
        _table("b", 30),
        _table("c", 10),
        _table("d", 10),
        # Pinned tables are walked in the order given rather than sorted, so
        # they belong here too: the node each lands on must not depend on it.
        _table("p", 10, pinned_to=1),
        _table("q", 10, pinned_to=0),
    ]
    first = BalancedNodePlacer().place(tables, _topology(), _nothing_spent())
    second = BalancedNodePlacer().place(
        list(reversed(tables)), _topology(), _nothing_spent()
    )
    assert first == second
    assert first["p"] == 1 and first["q"] == 0


def test_a_groupmate_with_spare_budget_makes_room():
    """What a table costs a node depends on what is already there.

    Two tables that fuse pool their HBM budgets and spend them on their pooled
    size, so one with budget to spare keeps the other out of host memory. Priced
    alone the second needs 14 GB of host and is refused; after pooling it needs
    none and fits. That is the review's second case -- host over-stated, a
    layout refused that the runtime would have held.
    """
    # Plenty of HBM, almost no host memory: the host figure is what decides.
    topo = _topology(hbm=80 * GB, ddr=4 * GB)
    generous = _table("generous", 4, budget_gb=20)  # 4 GB of a 20 GB budget
    needy = _table("needy", 16, budget_gb=2)  # 16 GB with 2 GB allowed

    assert needy.cost_alone().ddr == 14 * GB, "priced alone it spills 14 GB to host"
    assert generous.cost_alone().ddr == 0

    chosen = BalancedNodePlacer().place([generous, needy], topo, _nothing_spent())
    assert chosen["generous"] == chosen["needy"], (
        "they fuse, so together they fit in the pooled 22 GB budget and need no "
        "host memory -- which only shows if the placer prices them together"
    )


@pytest.mark.parametrize(
    "field, value, also_splits_the_dynamicemb_key",
    [
        # dist_type is in DynamicEmbTableOptions.get_grouped_key, so the
        # options themselves already differ.
        ("dist_type", "hash_roundrobin", True),
        # The rest are TorchRec's to group on and DynamicEmbTableOptions does
        # not carry them, so only the fusion key separates these tables:
        # the optimizer, whose state is part of a row and which TorchRec keys
        # on through the fused params...
        ("optimizer_type", OptimType.ADAM, False),
        # ...the pooling, which lives on the TorchRec config...
        ("pooling", PoolingType.SUM, False),
        # ...and the data type, which get_grouped_key leaves out because
        # DynamicEmb reads it off the options separately.
        ("embedding_dtype", torch.float16, False),
    ],
)
def test_tables_that_do_not_fuse_do_not_pool_their_budgets(
    field, value, also_splits_the_dynamicemb_key
):
    """Each parameter guards one component of `fusion_key`.

    The mirror of `test_a_groupmate_with_spare_budget_makes_room`: the same
    pair, separated on one field apiece, so they are priced on their own and
    `needy` spills to a host tier with no room for it. Drop the component being
    varied from `fusion_key` and the two fuse again, pool their budgets, need
    no host memory and fit -- the refusal disappears and this test fails.

    That is the unsafe direction, and the reason it is worth a test per
    component: a key coarser than the runtime's over-states HBM, which only
    refuses layouts that would have fit, but under-states the host tier by the
    same amount, which accepts layouts the runtime cannot hold (see
    `plan.fusion_key`).

    Two of the four also change what `needy` weighs -- Adam adds optimizer
    state to every row, float16 halves it, since `_table` sizes the row count
    as float32 -- which does not matter here: unpooled, all four overflow the
    4 GB host tier.
    """
    topo = _topology(hbm=80 * GB, ddr=4 * GB)
    generous = _table("generous", 4, budget_gb=20)
    needy = _table("needy", 16, budget_gb=2, **{field: value})

    # `DynamicEmbTableOptions.__eq__` compares `get_grouped_key()`, not its
    # fields, so options that differ only outside that key compare equal --
    # which is exactly why the fusion key has to carry more than they do.
    assert (generous.options != needy.options) is also_splits_the_dynamicemb_key
    assert generous.fusion_key != needy.fusion_key

    with pytest.raises(ValueError, match="No node can hold"):
        BalancedNodePlacer().place([generous, needy], topo, _nothing_spent())
