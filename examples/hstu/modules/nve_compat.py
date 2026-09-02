# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small compatibility helpers for the supported NVE Python generations."""

import os
import site
from pathlib import Path

_DEFAULT_NVE_INSTALL_ROOT = Path("/opt/nve")
_NVE_2605 = (26, 5)
_DEFAULT_NVE_MINIMUM = (26, 6)


def _expected_pynve_dir(version: str) -> Path:
    configured = os.environ.get(
        "NVE_INSTALL_ROOT", os.fspath(_DEFAULT_NVE_INSTALL_ROOT)
    )
    if not configured:
        raise RuntimeError("NVE_INSTALL_ROOT must not be empty")

    root = Path(configured).expanduser()
    if not root.is_absolute():
        raise RuntimeError(
            f"NVE_INSTALL_ROOT must be an absolute path, got {configured!r}"
        )
    install_root = root.resolve()
    site_roots = (
        Path(site_root).expanduser().resolve()
        for site_root in (*site.getsitepackages(), site.getusersitepackages())
        if site_root
    )
    containing_site_root = next(
        (
            site_root
            for site_root in site_roots
            if install_root.is_relative_to(site_root)
        ),
        None,
    )
    if containing_site_root is not None:
        raise RuntimeError(
            f"NVE_INSTALL_ROOT={install_root} is inside Python's automatically "
            f"searched site-packages root {containing_site_root}. Use an isolated "
            "root such as /opt/nve"
        )
    return (install_root / version / "python" / "pynve").resolve()


def _parse_generation(value: str, name: str) -> tuple[int, int]:
    parts = value.split(".")
    try:
        generation = (int(parts[0]), int(parts[1]))
    except (IndexError, ValueError) as error:
        raise RuntimeError(f"Invalid {name}={value!r}") from error
    if generation != _NVE_2605 and generation < _DEFAULT_NVE_MINIMUM:
        raise RuntimeError(
            f"Unsupported {name}={value!r}; expected 26.05 or 26.06 and later"
        )
    return generation


def imported_nve_generation() -> str:
    """Return the generation of the already selected ``pynve`` package.

    The development image deliberately has no unversioned pynve installation.
    A selected package must come from the ``26.05`` or ``default`` version.
    """
    import pynve

    version = str(getattr(pynve, "__version__", "unknown"))
    generation = _parse_generation(version, "pynve version")

    declared_version = os.environ.get("NVE_VERSION")
    declared_generation = (
        _parse_generation(declared_version, "NVE_VERSION")
        if declared_version
        else _DEFAULT_NVE_MINIMUM
    )
    selected_version = "26.05" if declared_generation == _NVE_2605 else "default"
    if (generation == _NVE_2605) != (selected_version == "26.05"):
        raise RuntimeError(
            f"NVE_VERSION={declared_version!r} selects {selected_version}, "
            "but imported "
            f"pynve {version} from {getattr(pynve, '__file__', None)}"
        )

    package_file = getattr(pynve, "__file__", None)
    if package_file is None:
        raise RuntimeError("The imported pynve package has no __file__")
    actual_dir = Path(package_file).resolve().parent
    expected_dir = _expected_pynve_dir(selected_version)
    if actual_dir != expected_dir:
        raise RuntimeError(
            f"pynve {version} was imported from {actual_dir}; expected "
            f"{expected_dir}. Select exactly one NVE Python prefix "
            "through PYTHONPATH"
        )
    return f"{generation[0]:02d}.{generation[1]:02d}"


def gpu_only_constructor_kwargs() -> dict[str, object]:
    """Return the generation-specific selector for a GPU-only NVE layer."""
    import pynve.torch.nve_layers as nve_layers

    if imported_nve_generation() == "26.05":
        return {"cache_type": nve_layers.CacheType.NoCache}
    return {"layer_type": nve_layers.LayerType.GPULayer}


def hierarchical_constructor_kwargs(storage: object) -> dict[str, object]:
    """Return the selector and backing store for a hierarchical NVE layer."""
    import pynve.torch.nve_layers as nve_layers

    if imported_nve_generation() == "26.05":
        return {
            "cache_type": nve_layers.CacheType.Hierarchical,
            "remote_interface": storage,
        }
    return {
        "layer_type": nve_layers.LayerType.Hierarchical,
        "storage": storage,
    }


def needs_legacy_embedding_lookup_fake_override() -> bool:
    """Return true only for the selected NVE 26.05 package."""
    try:
        return imported_nve_generation() == "26.05"
    except ModuleNotFoundError:
        return False
