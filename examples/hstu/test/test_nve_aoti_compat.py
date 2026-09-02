# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

HSTU_ROOT = Path(__file__).resolve().parents[1]
if str(HSTU_ROOT) not in sys.path:
    sys.path.insert(0, str(HSTU_ROOT))

from inference_aoti import nve_aoti_compat  # noqa: E402


def test_output_directories_are_created(tmp_path: Path) -> None:
    export_dir = tmp_path / "model"
    dump_dir = tmp_path / "replay"

    resolved_export, resolved_dump = nve_aoti_compat.prepare_output_directories(
        export_dir, dump_dir
    )

    assert Path(resolved_export) == export_dir.resolve()
    assert Path(resolved_dump) == dump_dir.resolve()
    assert export_dir.is_dir()
    assert dump_dir.is_dir()


def test_nonempty_output_is_rejected_before_creating_other_path(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "model"
    export_dir.mkdir()
    (export_dir / "existing").write_text("stale", encoding="utf-8")
    dump_dir = tmp_path / "replay"

    with pytest.raises(ValueError, match="output directory must be absent or empty"):
        nve_aoti_compat.prepare_output_directories(export_dir, dump_dir)
    assert not dump_dir.exists()


def test_nested_output_directories_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct, non-nested"):
        nve_aoti_compat.prepare_output_directories(
            tmp_path / "output", tmp_path / "output" / "replay"
        )
