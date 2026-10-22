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

"""pytest configuration for the dynamicemb tests.

* Everything under ``unit/`` is marked ``unit``, everything under
  ``integration/`` is marked ``integration``.
* A test file that must run on several ranks declares it once::

      DISTRIBUTED_NPROCS = (1, 8)

  and is then collected as one item per process count; each item re-runs the
  file under ``torchrun --nproc_per_node=N -m pytest``. The inner pytest
  collects the file normally, so a rank-aware test needs no launcher of its own.
  Fewer visible GPUs than N skips the item.
* ``current_device`` and ``backend_session`` are the fixtures every GPU test
  needs; a test file must use them rather than define its own copy.
* ``fwd_bwd`` / ``load_dump`` select the two CI groups (``unit_test.sh``);
  an item not marked ``load_dump`` is in ``fwd_bwd``.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from common.launch import WORKER_ENV, gpu_count, torchrun_command, worker_env

ROOT = Path(__file__).resolve().parent


def _is_worker() -> bool:
    # The launcher sets WORKER_ENV; a developer running torchrun by hand gets
    # torchrun's TORCHELASTIC_RUN_ID. LOCAL_RANK / WORLD_SIZE are not usable
    # for this: some container images export them in every shell.
    return WORKER_ENV in os.environ or "TORCHELASTIC_RUN_ID" in os.environ


def _declared_nprocs(path: Path):
    """The module-level ``DISTRIBUTED_NPROCS`` literal, read without importing."""
    try:
        tree = ast.parse(path.read_text())
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DISTRIBUTED_NPROCS"
            for t in node.targets
        ):
            return tuple(ast.literal_eval(node.value))
    return None


def _declared_marks(path: Path):
    """Names in a module-level ``pytestmark = pytest.mark.x`` (or a list of them)."""
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets
        ):
            return re.findall(r"pytest\.mark\.(\w+)", ast.unparse(node.value))
    return []


class DistributedFile(pytest.File):
    def collect(self):
        for nproc in _declared_nprocs(self.path):
            item = DistributedItem.from_parent(
                self, name=f"{self.path.name}[{nproc}-gpu]", nproc=nproc
            )
            # The outer item never imports the file, so its file-level marks
            # (e.g. load_dump) are read from the source and applied here.
            for mark in _declared_marks(self.path):
                item.add_marker(getattr(pytest.mark, mark))
            yield item


class DistributedItem(pytest.Item):
    def __init__(self, *, nproc: int, **kwargs):
        super().__init__(**kwargs)
        self.nproc = nproc

    def runtest(self):
        if gpu_count() < self.nproc:
            pytest.skip(f"needs {self.nproc} GPU(s), {gpu_count()} visible")
        opts = self.config.option
        cmd = torchrun_command(self.nproc, "-m", "pytest", str(self.path))
        cmd += ["-p", "no:cacheprovider", "-rs"]
        if opts.keyword:
            cmd += ["-k", opts.keyword]
        if opts.markexpr:
            cmd += ["-m", opts.markexpr]
        done = subprocess.run(cmd, env=worker_env(), cwd=ROOT.parent)
        if done.returncode == 5:  # the -k / -m selected nothing in this file
            pytest.skip("no test in the file matched the selection")
        assert done.returncode == 0, f"{' '.join(cmd)} exited {done.returncode}"

    def reportinfo(self):
        return self.path, None, self.name


def pytest_pycollect_makemodule(module_path, parent):
    # Replaces the Module for a file that declares DISTRIBUTED_NPROCS, so it is
    # not also collected (and run, on one rank) in this process.
    if not _is_worker() and _declared_nprocs(module_path):
        return DistributedFile.from_parent(parent, path=module_path)
    return None


def pytest_collection_modifyitems(items):
    for item in items:
        rel = Path(str(item.path)).resolve().relative_to(ROOT)
        if rel.parts[0] in ("unit", "integration"):
            item.add_marker(getattr(pytest.mark, rel.parts[0]))
        if item.get_closest_marker("load_dump") is None:
            item.add_marker(pytest.mark.fwd_bwd)


@pytest.fixture
def current_device():
    assert torch.cuda.is_available()
    return torch.cuda.current_device()


@pytest.fixture(scope="session")
def backend_session():
    """An NCCL process group over the ranks this pytest was started on.

    Under torchrun that is every rank; under a plain ``pytest`` it is a group of
    one, so a test that wants a process group does not care which it got.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29531")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    owned = not dist.is_initialized()
    if owned:
        dist.init_process_group(backend="nccl")
    dist.barrier()
    yield
    dist.barrier()
    if owned:
        dist.destroy_process_group()
