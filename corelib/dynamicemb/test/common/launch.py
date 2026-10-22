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

"""Run test code under ``torchrun``.

This is the only place that knows how a multi-process test is started, so a
test file declares *what* it needs (a process count, a set of worker flags) and
never spells a ``torchrun`` command line.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

import pytest

TEST_ROOT = Path(__file__).resolve().parent.parent

# Set by the launcher on the processes it starts; tells conftest.py that this
# pytest is itself a worker and must collect the file instead of relaunching it.
WORKER_ENV = "DYNAMICEMB_TEST_WORKER"


def gpu_count() -> int:
    import torch

    return torch.cuda.device_count()


def require_gpus(n: int) -> None:
    """Skip the calling test when fewer than ``n`` GPUs are visible."""
    if gpu_count() < n:
        pytest.skip(f"needs {n} GPU(s), {gpu_count()} visible")


def torchrun_command(nproc: int, *target: str) -> list:
    """``torchrun --standalone`` over ``target`` (a script, or ``-m module ...``)."""
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={nproc}",
        *target,
    ]


def worker_env(extra: Optional[Mapping[str, str]] = None) -> dict:
    env = dict(os.environ)
    env[WORKER_ENV] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(TEST_ROOT), env.get("PYTHONPATH", "")) if p
    )
    env.update(extra or {})
    return env


def run_worker(
    script: Path,
    nproc: int,
    args: Sequence[str] = (),
    env: Optional[Mapping[str, str]] = None,
) -> None:
    """Run a worker script (an argparse ``main``) on ``nproc`` ranks; raise on failure.

    Skips, rather than fails, when the machine has fewer than ``nproc`` GPUs.
    """
    require_gpus(nproc)
    cmd = torchrun_command(nproc, str(script), *map(str, args))
    done = subprocess.run(cmd, env=worker_env(env), cwd=TEST_ROOT.parent)
    assert done.returncode == 0, f"worker exited {done.returncode}: {' '.join(cmd)}"
