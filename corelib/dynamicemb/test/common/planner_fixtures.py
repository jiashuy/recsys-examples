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

"""Machine descriptions shared by the planner unit tests.

The placement tests and the plan-cost tests both start from a TorchRec
``Topology`` that is pure arithmetic (no GPU, no process group); only the world
shape differs between them, so the shape is an argument here.
"""

from torchrec.distributed.planner.types import Topology

GB = 1024**3


def make_topology(
    world_size: int,
    local_world_size: int,
    hbm: int = 80 * GB,
    ddr: int = 128 * GB,
) -> Topology:
    """A homogeneous machine: every rank has `hbm` bytes of HBM and `ddr` of host."""
    return Topology(
        world_size=world_size,
        local_world_size=local_world_size,
        compute_device="cuda",
        hbm_cap=hbm,
        ddr_cap=ddr,
    )
