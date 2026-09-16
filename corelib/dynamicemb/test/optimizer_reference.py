# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Pure-torch optimizer steps to compare the fused kernels against.

Most optimizers dynamicemb implements have an FBGEMM counterpart, so the tests
check them against a ``SplitTableBatchedEmbeddingBagsCodegen`` built with the
same hyperparameters. FTRL has no such counterpart, and a second dynamicemb
storage is no use as a reference -- it shares the very optimizer under test, so
a wrong formula would come out wrong on both sides. What follows is an
independent transcription instead.

Kept out of any ``test_*`` module so pytest does not collect it, and computed in
float64 so it stays a reference rather than a second copy of the kernel's
rounding.
"""

from typing import Tuple

import torch


def ftrl_step(
    weight: torch.Tensor,
    linear: torch.Tensor,
    accum: torch.Tensor,
    grad: torch.Tensor,
    learning_rate: float,
    learning_rate_power: float = -0.5,
    ftrl_beta: float = 0.0,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One FTRL-Proximal step, from Algorithm 1 of McMahan et al. 2013.

    ``learning_rate``, ``ftrl_beta``, ``l1_reg`` and ``l2_reg`` are the paper's
    alpha, beta, lambda1 and lambda2. The paper fixes the accumulator's exponent
    at a square root; ``learning_rate_power`` generalizes it, and its default of
    -0.5 recovers the paper.

    Returns the new ``(weight, linear, accum)``. Elementwise, so it takes any
    shape as long as all four arguments agree.
    """
    weight = weight.to(torch.float64)
    linear = linear.to(torch.float64)
    accum = accum.to(torch.float64)
    grad = grad.to(torch.float64)

    exponent = -learning_rate_power
    new_accum = accum + grad * grad
    new_accum_pow = new_accum**exponent

    linear = (
        linear
        + grad
        - (new_accum_pow - accum**exponent) / learning_rate * weight
    )
    shrunk = (l1_reg * torch.sign(linear) - linear) / (
        (ftrl_beta + new_accum_pow) / learning_rate + l2_reg
    )
    # |linear| <= l1_reg pins the weight to exactly zero; that is the whole
    # point of the L1 term and what separates FTRL from a shrinkage update.
    weight = torch.where(linear.abs() > l1_reg, shrunk, torch.zeros_like(shrunk))
    return weight, linear, new_accum
