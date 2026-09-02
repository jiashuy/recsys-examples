# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared output-path and NVE AOTI loading helpers for the exporters."""

import os
from pathlib import Path
from typing import Any

import torch
from modules.nve_compat import imported_nve_generation


def prepare_output_directories(
    export_dir: str | os.PathLike[str],
    dump_dir: str | os.PathLike[str],
) -> tuple[str, str]:
    """Create two distinct, empty exporter destinations."""
    export_path = Path(export_dir).resolve()
    dump_path = Path(dump_dir).resolve()
    paths = (export_path, dump_path)

    if (
        export_path == dump_path
        or export_path in dump_path.parents
        or dump_path in export_path.parents
    ):
        raise ValueError(
            "--export_dir and --dump_dir must be distinct, non-nested paths"
        )

    for path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise ValueError(f"output directory must be absent or empty: {path}")

    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    return str(export_path), str(dump_path)


def load_aoti(
    package_dir: str | os.PathLike[str],
    device: torch.device,
) -> tuple[Any, list[Any]]:
    """Load an AOTI model with the selected NVE generation."""
    package_dir = os.fspath(package_dir)
    if imported_nve_generation() != "26.05":
        from pynve.torch.nve_export import load_aot

        return load_aot(package_dir, device=device)

    from pynve.torch.nve_export import load_nve_layers
    from torch._C._aoti import AOTIModelPackageLoader

    if device.type != "cuda":
        raise ValueError("NVE 26.05 AOTI loading requires a CUDA device")
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    with torch.cuda.device(device_index):
        nve_layers = load_nve_layers(package_dir)
    aoti_model_runtime = AOTIModelPackageLoader(
        os.path.join(package_dir, "model.pt2"),
        "model",
        False,
        1,
        device_index,
    )
    return aoti_model_runtime, nve_layers
