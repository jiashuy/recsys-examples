#!/usr/bin/env python3
# Idempotent in-place patches for Blackwell (sm_10+).
#
# Two classes of patches:
#
#   1) Guard `import hstu.hstu_ops_gpu` in fused_hstu_op.py:
#      register_fake('fbgemm::hstu_varlen_fwd_80') fails at import time
#      because the C++ op is not shipped for sm_10.
#
#   2) Widen hard-coded SM-version checks `elif sm == 9` (and the
#      equivalent `elif sm_major_version == 9`) to `elif sm >= 9`,
#      so Blackwell falls into the Hopper path instead of `raise`.
#      Affects fused_hstu_op.py (4 sites) and paged_hstu_infer_layer.py.
#
# Idempotent: each patch is gated on a marker substring so re-runs are
# no-ops. Files outside the listed target basenames are ignored.

import pathlib
import sys

# (target_basename, [(old, new, marker_that_indicates_already_patched), ...])
RULES = {
    "fused_hstu_op.py": [
        (
            "import hstu.hstu_ops_gpu  # noqa: F401 – registers fake impls for torch.export",
            (
                "import torch as _t\n"
                "if _t.cuda.is_available() and _t.cuda.get_device_capability() < (10, 0):\n"
                "    import hstu.hstu_ops_gpu  # noqa: F401  # registers fake impls for torch.export"
            ),
            "get_device_capability",
        ),
        # Widen the addmm/silu dispatch (uses bare `sm`) so Blackwell falls
        # into the Hopper branch which uses torch_addmm — works on cuBLAS.
        (
            "elif sm == 9:",
            "elif sm >= 9:",
            "elif sm >= 9:",
        ),
        # NOTE: do NOT auto-widen `elif sm_major_version == 9:` — those are
        # the HSTU attention dispatch sites and the SM9 branch calls
        # fbgemm.hstu_varlen_fwd_90/bwd_90, which the Blackwell container
        # does not ship. The host-side source already adds an explicit
        # `elif sm_major_version >= 10:` branch that routes Blackwell to
        # hstu.hstu_attn_varlen_func. Widening here would silently funnel
        # Blackwell back into the broken fbgemm sm90 path.
    ],
    "paged_hstu_infer_layer.py": [
        (
            "elif sm == 9:",
            "elif sm >= 9:",
            "elif sm >= 9:",
        ),
    ],
}

DEFAULT_ROOTS = [
    "/Workspace",
    "/workspace",
    "/opt",
    "/usr/local/lib",
    "/root",
]


def patch_file(p: pathlib.Path, rules) -> int:
    """Return number of edits applied to *p*."""
    try:
        s = p.read_text()
    except (OSError, UnicodeDecodeError):
        return 0
    edits = 0
    for old, new, marker in rules:
        # Skip if file shows no sign of the old pattern but already has marker,
        # or if old isn't in the file at all.
        if old not in s:
            continue
        # If marker is identical to `new`, that's fine: replacement is idempotent
        # because s.replace(old, new) won't match `new` content.
        if marker != new and marker in s:
            continue
        new_s = s.replace(old, new)
        if new_s != s:
            s = new_s
            edits += 1
    if edits:
        p.write_text(s)
        print(f"[blackwell-patch] {p}  ({edits} edit(s))")
    return edits


roots = sys.argv[1:] or DEFAULT_ROOTS
total_files = 0
total_edits = 0
for r in roots:
    p_root = pathlib.Path(r)
    if not p_root.exists():
        continue
    for basename, rules in RULES.items():
        for f in p_root.rglob(basename):
            n = patch_file(f, rules)
            if n:
                total_files += 1
                total_edits += n

print(f"[blackwell-patch] patched {total_edits} edit(s) across {total_files} file(s)")
