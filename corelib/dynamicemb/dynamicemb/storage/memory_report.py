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

"""The memory a module's tables take, per tier, printed once at construction."""

from typing import List, Optional, Tuple

from dynamicemb.dynamicemb_config import DynamicEmbTableOptions
from dynamicemb.optimizer import (
    BaseDynamicEmbeddingOptimizer,
    OptimType,
    get_optimizer_state_dim,
)
from dynamicemb.storage.storage_layout import StorageLayout
from dynamicemb.types import Cache, Storage
from dynamicemb.utils import DTYPE_NUM_BYTES

def _tier_bytes(
    values, options_list: Optional[List[DynamicEmbTableOptions]] = None
) -> List[Tuple[int, int, int, int, int, int]]:
    """Per-table value bytes for a built tier: (init total/emb/opt, peak total/emb/opt).

    ``init`` is the current ``ExtendableBuffer`` logical size (post bucket-align,
    pre-rehash). ``peak`` is ``max_capacity * value_dim * element_size`` — what
    the buffer grows to after rehashing fully.
    """
    rows = []
    for t in range(values.num_tables):
        buf = values.buffers[t]
        value_dim = values.value_dim(t)
        emb_dim = values.emb_dim(t)
        elem = DTYPE_NUM_BYTES[values.dtype]
        init_total = buf.num_bytes()
        init_emb = init_total * emb_dim // value_dim
        init_opt = init_total - init_emb
        # A buffer that does not grow is as big at its peak as it is now.
        peak_total = (
            init_total
            if options_list is None
            else options_list[t].max_capacity * value_dim * elem
        )
        peak_emb = peak_total * emb_dim // value_dim
        peak_opt = peak_total - peak_emb
        rows.append((init_total, init_emb, init_opt, peak_total, peak_emb, peak_opt))
    return rows


def _tier_bytes_of(state) -> List[Tuple[int, int, int, int, int, int]]:
    return _tier_bytes(state.values, state.options_list)


def _tier_bytes_from_options(
    options_list: List[DynamicEmbTableOptions],
    optimizer: Optional["BaseDynamicEmbeddingOptimizer"],
    emb_optimizer_type: OptimType,
) -> List[Tuple[Optional[int], Optional[int], Optional[int], int, int, int]]:
    """Fallback for external-PS tiers where we can't peek into the storage.

    Init bytes are unknown (PS is opaque) — reported as ``None``. Peak uses the
    original options' ``max_capacity``.
    """
    rows = []
    for opt in options_list:
        emb_dim = opt.dim
        if optimizer is not None:
            optim_state_dim = optimizer.get_state_dim(emb_dim)
        else:
            optim_state_dim = get_optimizer_state_dim(
                emb_optimizer_type, emb_dim, opt.embedding_dtype
            )
        value_dim = emb_dim + optim_state_dim
        elem = DTYPE_NUM_BYTES[opt.embedding_dtype]
        peak_total = opt.max_capacity * value_dim * elem
        peak_emb = peak_total * emb_dim // value_dim
        peak_opt = peak_total - peak_emb
        rows.append((None, None, None, peak_total, peak_emb, peak_opt))
    return rows


def _build_tier_block(
    tier_label: str,
    tier_rows: List[Tuple],
    table_names: List[str],
    fmt,
) -> Tuple[List[List[str]], List[str]]:
    """Per-tier long-format rows + per-tier TOTAL row.

    ``tier_label`` is emitted only on the first row; later rows leave that cell
    blank so the tier column visually groups its tables. TOTAL aggregates
    init/peak/emb/opt; init becomes ``N/A`` if any contributing row's init is
    unknown (external PS).
    """
    rows: List[List[str]] = []
    init_known = True
    sum_init = sum_pt = sum_pe = sum_po = 0
    for ti, (it, _ie, _io, pt, pe, po) in enumerate(tier_rows):
        rows.append(
            [
                tier_label if ti == 0 else "",
                table_names[ti],
                fmt(it),
                fmt(pt),
                f"{fmt(pe)} / {fmt(po)}",
            ]
        )
        if it is None:
            init_known = False
        else:
            sum_init += it
        sum_pt += pt
        sum_pe += pe
        sum_po += po
    total = [
        "",
        "TOTAL",
        fmt(sum_init) if init_known else "N/A",
        fmt(sum_pt),
        f"{fmt(sum_pe)} / {fmt(sum_po)}",
    ]
    return rows, total


def _render_tier_table(
    header: List[str],
    tier_blocks: List[Tuple[List[List[str]], List[str]]],
) -> str:
    """Render the long-format table.

    Two kinds of horizontal rules:
      * ``sep_full`` — top border, under the header, between tier blocks, and
        the closing bottom border.
      * ``sep_blank_tier`` — frames each TOTAL row (tier column left blank so
        the rule reads as part of the TOTAL row, not as a tier boundary).
    """
    all_rows: List[List[str]] = [header]
    for table_rows, total_row in tier_blocks:
        all_rows.extend(table_rows)
        all_rows.append(total_row)
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(header))]

    def fmt_row(cells: List[str], center: bool = False) -> str:
        return " | ".join(
            (c.center(w) if center else c.ljust(w)) for c, w in zip(cells, widths)
        )

    sep_full = "-+-".join("-" * w for w in widths)
    sep_blank_tier = " " * widths[0] + " | " + "-+-".join("-" * w for w in widths[1:])

    lines = [sep_full, fmt_row(header, center=True), sep_full]
    last = len(tier_blocks) - 1
    for i, (table_rows, total_row) in enumerate(tier_blocks):
        if i > 0:
            lines.append(sep_full)
        for r in table_rows:
            lines.append(fmt_row(r))
        lines.append(sep_blank_tier)
        lines.append(fmt_row(total_row))
        if i == last:
            lines.append(sep_full)
    return "\n".join(lines)


def print_memory_consume(
    table_names: List[str],
    storage: "Storage",
    cache: Optional["Cache"],
    layout: StorageLayout,
    dynamicemb_options: List[DynamicEmbTableOptions],
    optimizer: Optional["BaseDynamicEmbeddingOptimizer"],
    device_id: int,
    emb_optimizer_type: OptimType,
) -> None:
    """Value-only memory accounting (embedding + optimizer state rows).

    The caller passes the ``StorageLayout`` set by ``_create_cache_storage`` so
    the report matches the code path that ran (no inference from buffer state).
    Per-tier bytes come from the constructed ``ExtendableBuffer`` (init) and
    ``options.max_capacity`` (peak), capturing bucket alignment, hybrid
    ``cap_scale`` rebalancing, and caching's HBM/backing duplication. Does
    *not* include hash-table metadata (keys/digests/scores/ref_counter), which
    always live on HBM regardless of value tier.
    """
    # Build (tier_label, per-table rows) list. Each row is the 6-tuple above.
    tiers: List[Tuple[str, List[Tuple]]] = []
    if layout == StorageLayout.HBM_ONLY:
        tiers.append((f"hbm/cuda:{device_id}", _tier_bytes_of(storage._state)))
    elif layout == StorageLayout.HOST_ONLY:
        tiers.append(("host (CPU)", _tier_bytes_of(storage._state)))
    elif layout == StorageLayout.HYBRID:
        tiers.append((f"hbm/cuda:{device_id}", _tier_bytes_of(storage._hbm)))
        tiers.append(("host (CPU)", _tier_bytes_of(storage._host)))
    elif layout == StorageLayout.CACHING:
        tiers.append(
            (f"hbm cache/cuda:{device_id}", _tier_bytes(cache.values))
        )
        tiers.append(("host backing (full)", _tier_bytes_of(storage._state)))
    else:
        raise ValueError(f"Unhandled StorageLayout: {layout!r}")
    if getattr(storage, "lower", None) is not None:
        # Below the tables, taking up what they evict. Opaque: no sizes to read.
        tiers.append(
            (
                "external PS",
                _tier_bytes_from_options(
                    dynamicemb_options, optimizer, emb_optimizer_type
                ),
            )
        )

    # Pick a single unit (MB if any value reaches 1 MB; else KB).
    max_byte = 0
    for _, rows in tiers:
        for row in rows:
            for v in row:
                if v is not None and v > max_byte:
                    max_byte = v
    use_mb = max_byte >= 1024 * 1024
    unit = "MB" if use_mb else "KB"
    divisor = 1024 * 1024 if use_mb else 1024

    def F(x):
        if x is None:
            return "N/A"
        # Up to 3 decimals; drop trailing zeros so whole values render as ints.
        s = f"{x / divisor:.3f}".rstrip("0").rstrip(".")
        return s or "0"

    header = [
        "tier",
        "table",
        f"init({unit})",
        f"peak({unit})",
        f"emb/opt peak({unit})",
    ]
    tier_blocks = [
        _build_tier_block(label, rows, table_names, F) for label, rows in tiers
    ]
    body = _render_tier_table(header, tier_blocks)
    output = f"\n\nStorage layout: {layout.value}\n" + body
    if layout == StorageLayout.CACHING:
        output += (
            "\nNote: HBM cache rows duplicate backing rows; "
            "total memory ≈ HBM cache + Host backing (no subtract)."
        )
    print(output)


