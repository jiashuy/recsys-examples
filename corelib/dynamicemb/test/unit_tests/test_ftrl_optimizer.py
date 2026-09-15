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

"""FTRL is checked against Algorithm 1 of McMahan et al. 2013.

The value layout under test is ``[embedding | linear | accum]``, each region
``emb_dim`` wide, matching what :func:`get_optimizer_state_dim` reserves.
"""

from typing import Tuple

import pytest
import torch
from dynamicemb.optimizer import (
    DynamicEmbOptimType,
    FTRLDynamicEmbeddingOptimizer,
    OptimizerArgs,
    get_optimizer_ckpt_state_dim,
    get_optimizer_state_dim,
)


def _make_optimizer(
    learning_rate: float = 0.1,
    learning_rate_power: float = -0.5,
    initial_accumulator_value: float = 0.0,
    ftrl_beta: float = 0.0,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
) -> FTRLDynamicEmbeddingOptimizer:
    return FTRLDynamicEmbeddingOptimizer(
        OptimizerArgs(
            learning_rate=learning_rate,
            learning_rate_power=learning_rate_power,
            initial_accumulator_value=initial_accumulator_value,
            ftrl_beta=ftrl_beta,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
        )
    )


def _ftrl_reference(
    weight: torch.Tensor,
    linear: torch.Tensor,
    accum: torch.Tensor,
    grad: torch.Tensor,
    learning_rate: float,
    learning_rate_power: float,
    ftrl_beta: float,
    l1_reg: float,
    l2_reg: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One FTRL step, transcribed from Algorithm 1 of McMahan et al. 2013.

    Computed in float64 so it stays a reference rather than a second copy of the
    kernel's rounding.
    """
    weight = weight.to(torch.float64)
    linear = linear.to(torch.float64)
    accum = accum.to(torch.float64)
    grad = grad.to(torch.float64)

    new_accum = accum + grad * grad
    power = -learning_rate_power
    new_accum_pow = new_accum**power
    accum_pow = accum**power

    linear = linear + grad - (new_accum_pow - accum_pow) / learning_rate * weight
    shrunk = (l1_reg * torch.sign(linear) - linear) / (
        (ftrl_beta + new_accum_pow) / learning_rate + l2_reg
    )
    weight = torch.where(linear.abs() > l1_reg, shrunk, torch.zeros_like(shrunk))
    return weight, linear, new_accum


def _split(values: torch.Tensor, emb_dim: int, state_offset: int):
    """Split a fused value row into its embedding / linear / accum regions."""
    weight = values[:, :emb_dim]
    linear = values[:, state_offset : state_offset + emb_dim]
    accum = values[:, state_offset + emb_dim : state_offset + 2 * emb_dim]
    return weight, linear, accum


def _fresh_values(
    num_rows: int,
    emb_dim: int,
    row_width: int,
    state_offset: int,
    accum_seed: float,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.zeros(num_rows, row_width, dtype=torch.float32)
    values[:, :emb_dim] = torch.rand(
        num_rows, emb_dim, generator=generator, dtype=torch.float32
    )
    values[:, state_offset : state_offset + emb_dim] = 0.0
    values[
        :, state_offset + emb_dim : state_offset + 2 * emb_dim
    ] = accum_seed
    return values.to(device)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="FTRL kernels are CUDA-only"
)


def test_state_dim_is_two_regions():
    for dim in (1, 4, 5, 128):
        assert get_optimizer_state_dim(DynamicEmbOptimType.FTRL, dim) == 2 * dim
        # FTRL keeps its full state in checkpoints; nothing is reconstructible.
        assert get_optimizer_ckpt_state_dim(DynamicEmbOptimType.FTRL, dim) == 2 * dim
        assert _make_optimizer().get_state_dim(dim) == 2 * dim


def test_reset_seeds_accum_only():
    """``linear`` starts at zero, ``accum`` at initial_accumulator_value -- z
    and n of the paper, which begin at 0 and at whatever seeds the learning
    rate."""
    optimizer = _make_optimizer(initial_accumulator_value=0.25)
    states = torch.full((2, 6), 7.0)
    optimizer.reset_optimizer_states(states, emb_dims=3)
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.25, 0.25, 0.25]] * 2)
    torch.testing.assert_close(states, expected)


def test_reset_with_a_zero_seed_needs_no_widths():
    """Both halves are zero, so the uniform fill is exact and emb_dims is moot."""
    optimizer = _make_optimizer(initial_accumulator_value=0.0)
    assert optimizer.get_initial_optimizer_state() == 0.0
    states = torch.full((2, 8), 7.0)
    optimizer.reset_optimizer_states(states)
    torch.testing.assert_close(states, torch.zeros(2, 8))


def test_reset_rejects_a_non_zero_seed_without_widths():
    # The state width alone cannot locate the boundary in a padded buffer.
    optimizer = _make_optimizer(initial_accumulator_value=0.25)
    with pytest.raises(ValueError, match="emb_dims"):
        optimizer.reset_optimizer_states(torch.empty(2, 6))


def test_reset_honours_per_row_widths():
    """A padded buffer reserves the widest table's state for every row, so a
    narrower table's rows split earlier and leave slack columns behind."""
    optimizer = _make_optimizer(initial_accumulator_value=0.25)
    states = torch.full((2, 6), 7.0)
    optimizer.reset_optimizer_states(states, emb_dims=torch.tensor([3, 2]))
    torch.testing.assert_close(
        states[0], torch.tensor([0.0, 0.0, 0.0, 0.25, 0.25, 0.25])
    )
    # Row 1 is a dim-2 table: linear, accum, then two columns of slack.
    torch.testing.assert_close(states[1], torch.tensor([0.0, 0.0, 0.25, 0.25, 0.0, 0.0]))


def test_reset_writes_only_the_selected_rows():
    optimizer = _make_optimizer(initial_accumulator_value=0.25)
    states = torch.full((3, 4), 7.0)
    optimizer.reset_optimizer_states(
        states, indices=torch.tensor([2]), emb_dims=torch.tensor([2])
    )
    torch.testing.assert_close(states[:2], torch.full((2, 4), 7.0))
    torch.testing.assert_close(states[2], torch.tensor([0.0, 0.0, 0.25, 0.25]))


@requires_cuda
def test_matches_hand_computed_closed_form():
    """Pins the kernel to arithmetic a reader can redo by hand.

    With l1 = l2 = beta = 0 the closed form collapses to

        w_next = w * (1 - n**0.5 / n_next**0.5) - lr * g / n_next**0.5

    so for w=1, lr=0.1, n=0.1 and g=1: 1 * (1 - 0.3162278/1.0488088)
    - 0.1/1.0488088 = 0.6985 - 0.09535 = 0.6031424. The second row runs the
    same arithmetic at g=2, and the expectations below are both steps of both.
    """
    device = torch.device("cuda")
    emb_dim = 1
    row_width = 3 * emb_dim
    optimizer = _make_optimizer()

    values = torch.zeros(2, row_width, dtype=torch.float32, device=device)
    values[:, 0] = 1.0
    values[:, 1] = 0.0
    values[:, 2] = 0.1
    grads = torch.tensor([[1.0], [2.0]], dtype=torch.float32, device=device)
    table_ids = torch.zeros(2, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor([emb_dim], dtype=torch.int64, device=device)

    optimizer.update_for_padded_buffer(
        grads, values, table_ids, table_emb_dims, emb_dim, row_width, False
    )
    torch.testing.assert_close(
        values[:, 0].cpu(),
        torch.tensor([0.6031424, 0.7450533]),
        rtol=0,
        atol=1e-6,
    )

    optimizer.update_for_padded_buffer(
        grads, values, table_ids, table_emb_dims, emb_dim, row_width, False
    )
    torch.testing.assert_close(
        values[:, 0].cpu(),
        torch.tensor([0.5341358, 0.6747804]),
        rtol=0,
        atol=1e-6,
    )


@requires_cuda
@pytest.mark.parametrize("emb_dim,all_dims_vec4", [(4, True), (5, False), (128, True)])
@pytest.mark.parametrize("learning_rate_power", [-0.5, -0.25])
@pytest.mark.parametrize("ftrl_beta", [0.0, 1.0])
@pytest.mark.parametrize("l1_reg,l2_reg", [(0.0, 0.0), (0.02, 0.0), (0.02, 0.05)])
def test_padded_buffer_matches_reference(
    emb_dim, all_dims_vec4, learning_rate_power, ftrl_beta, l1_reg, l2_reg
):
    device = torch.device("cuda")
    num_rows = 64
    row_width = 3 * emb_dim
    accum_seed = 0.1
    optimizer = _make_optimizer(
        learning_rate_power=learning_rate_power,
        ftrl_beta=ftrl_beta,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
    )

    values = _fresh_values(
        num_rows, emb_dim, row_width, emb_dim, accum_seed, device
    )
    table_ids = torch.zeros(num_rows, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor([emb_dim], dtype=torch.int64, device=device)

    weight, linear, accum = (t.clone() for t in _split(values, emb_dim, emb_dim))
    generator = torch.Generator(device="cpu").manual_seed(7)

    for _ in range(3):
        grads = (
            torch.randn(num_rows, emb_dim, generator=generator, dtype=torch.float32).to(
                device
            )
            * 0.5
        )
        optimizer.update_for_padded_buffer(
            grads, values, table_ids, table_emb_dims, emb_dim, row_width, all_dims_vec4
        )
        weight, linear, accum = _ftrl_reference(
            weight,
            linear,
            accum,
            grads,
            optimizer._opt_args.learning_rate,
            learning_rate_power,
            ftrl_beta,
            l1_reg,
            l2_reg,
        )

    got_weight, got_linear, got_accum = _split(values, emb_dim, emb_dim)
    torch.testing.assert_close(
        got_weight.double(), weight, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(
        got_linear.double(), linear, rtol=1e-4, atol=1e-5
    )
    torch.testing.assert_close(got_accum.double(), accum, rtol=1e-5, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("emb_dim,all_dims_vec4", [(4, True), (5, False)])
def test_flat_table_matches_padded_buffer(emb_dim, all_dims_vec4):
    """The two kernels differ only in how they find a row, not in the update."""
    device = torch.device("cuda")
    num_rows = 32
    row_width = 3 * emb_dim
    accum_seed = 0.1
    optimizer = _make_optimizer()

    padded = _fresh_values(
        num_rows, emb_dim, row_width, emb_dim, accum_seed, device, seed=3
    )
    flat = padded.clone()
    generator = torch.Generator(device="cpu").manual_seed(11)
    grads = torch.randn(
        num_rows, emb_dim, generator=generator, dtype=torch.float32
    ).to(device)

    table_ids = torch.zeros(num_rows, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor([emb_dim], dtype=torch.int64, device=device)
    table_value_dims = torch.tensor([row_width], dtype=torch.int64, device=device)

    optimizer.update_for_padded_buffer(
        grads, padded, table_ids, table_emb_dims, emb_dim, row_width, all_dims_vec4
    )

    optimizer.fused_update_for_flat_table(
        grads,
        torch.arange(num_rows, dtype=torch.int64, device=device),
        torch.tensor([flat.data_ptr()], dtype=torch.int64, device=device),
        table_ids,
        table_value_dims,
        table_emb_dims,
        emb_dim,
        all_dims_vec4,
        torch.float32,
    )

    torch.testing.assert_close(flat, padded, rtol=0, atol=0)


@requires_cuda
def test_l1_drives_weights_to_exact_zero():
    """The point of FTRL: L1 yields real zeros, not merely small weights."""
    device = torch.device("cuda")
    emb_dim = 4
    row_width = 3 * emb_dim
    optimizer = _make_optimizer(l1_reg=5.0)

    values = _fresh_values(64, emb_dim, row_width, emb_dim, 0.1, device, seed=5)
    table_ids = torch.zeros(64, dtype=torch.int64, device=device)
    table_emb_dims = torch.tensor([emb_dim], dtype=torch.int64, device=device)
    grads = torch.full((64, emb_dim), 0.01, dtype=torch.float32, device=device)

    optimizer.update_for_padded_buffer(
        grads, values, table_ids, table_emb_dims, emb_dim, row_width, True
    )

    weight, _, _ = _split(values, emb_dim, emb_dim)
    assert bool((weight == 0.0).all()), weight


@requires_cuda
def test_skipped_rows_are_untouched():
    """-1 marks a key that failed to insert; its row must not be written."""
    device = torch.device("cuda")
    emb_dim = 4
    row_width = 3 * emb_dim
    optimizer = _make_optimizer()

    values = _fresh_values(8, emb_dim, row_width, emb_dim, 0.1, device, seed=9)
    before = values.clone()
    indices = torch.full((8,), -1, dtype=torch.int64, device=device)
    indices[3] = 3
    grads = torch.ones(8, emb_dim, dtype=torch.float32, device=device)

    optimizer.fused_update_for_flat_table(
        grads,
        indices,
        torch.tensor([values.data_ptr()], dtype=torch.int64, device=device),
        torch.zeros(8, dtype=torch.int64, device=device),
        torch.tensor([row_width], dtype=torch.int64, device=device),
        torch.tensor([emb_dim], dtype=torch.int64, device=device),
        emb_dim,
        True,
        torch.float32,
    )

    untouched = [r for r in range(8) if r != 3]
    torch.testing.assert_close(values[untouched], before[untouched], rtol=0, atol=0)
    assert not bool(torch.equal(values[3], before[3]))


def test_non_positive_learning_rate_is_rejected():
    # FTRL divides by the learning rate; a zero would be a silent inf/nan.
    with pytest.raises(ValueError, match="must be positive"):
        _make_optimizer(learning_rate=0.0)


def test_positive_learning_rate_power_is_rejected():
    """The accumulator is raised to -learning_rate_power, so a positive value
    puts it in the denominator and the learning rate grows without bound."""
    with pytest.raises(ValueError, match="must be <= 0"):
        _make_optimizer(learning_rate_power=0.5)


def test_zero_learning_rate_power_is_allowed():
    # accum**0 == 1, i.e. a fixed learning rate -- degenerate but meaningful.
    assert _make_optimizer(learning_rate_power=0.0) is not None


def test_checkpoint_args_are_held_to_the_same_bounds():
    """``set_opt_args`` takes a checkpoint's meta, which nothing else vets."""
    optimizer = _make_optimizer()
    meta = dict(optimizer.get_opt_args())
    meta["learning_rate_power"] = 0.5
    with pytest.raises(ValueError, match="must be <= 0"):
        optimizer.set_opt_args(meta)
