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

"""Bandwidth and HBM figures of the GPU a test runs on, for building a planner Topology."""

import torch


class Platform:
    """Interconnect bandwidth and HBM capacity of the A100 / H100 / H200 a rank is on."""

    def __init__(self, device):
        device_id = device.index
        gpu_name = torch.cuda.get_device_name(device_id)
        if "A100" in gpu_name:
            self.platform = "a100"
            self.intra_host_bw = 300e9
            self.inter_host_bw = 25e9
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H100" in gpu_name:
            self.platform = "h100"
            self.intra_host_bw = 450e9
            self.inter_host_bw = 25e9  # TODO: need check
            self.hbm_cap = 80 * 1024 * 1024 * 1024
        elif "H200" in gpu_name:
            self.platform = "h200"
            self.intra_host_bw = 450e9
            self.inter_host_bw = 450e9
            self.hbm_cap = 140 * 1024 * 1024 * 1024
        else:
            raise RuntimeError(f"Not plan for {gpu_name}")
