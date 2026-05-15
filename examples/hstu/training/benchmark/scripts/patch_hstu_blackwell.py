#!/usr/bin/env python3
# Idempotent in-place patches for Blackwell (sm_10+).
#
# Three classes of patches:
#
#   1) Guard `import hstu.hstu_ops_gpu` in fused_hstu_op.py:
#      register_fake('fbgemm::hstu_varlen_fwd_80') fails at import time
#      because the C++ op is not shipped for sm_10.
#
#   2) Widen hard-coded SM-version checks `elif sm == 9` to `elif sm >= 9`
#      so Blackwell falls into the Hopper path for the addmm/silu dispatch
#      (cuBLAS works on Blackwell). NOTE: we deliberately do NOT widen the
#      `sm_major_version == 9` HSTU attention sites — those call fbgemm
#      sm90 ops that aren't shipped for Blackwell; the host-side source
#      handles SM10 explicitly with hstu.hstu_attn_varlen_func.
#
#   3) Rewrite `fbgemm_gpu.experimental.hstu.hstu_blackwell` to
#      `hstu.hstu_blackwell` in any .py file. The installed hstu package
#      points its Blackwell branch at fbgemm_gpu.experimental, but the
#      installed fbgemm_gpu has no `experimental/` subpackage. The actual
#      Blackwell Triton code is already shipped under hstu/hstu_blackwell/,
#      so we redirect every import path that hits the missing namespace.
#
# Idempotent: each patch is gated on a marker substring so re-runs are
# no-ops.

import pathlib
import sys

# Per-file rules: applied only when scanning files whose basename matches.
# Format: (old, new, marker_that_indicates_already_patched)
PER_FILE_RULES = {
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
        # does not ship. The host-side source adds an explicit
        # `elif sm_major_version >= 10:` branch that routes Blackwell to
        # hstu.hstu_attn_varlen_func.
    ],
    "paged_hstu_infer_layer.py": [
        (
            "elif sm == 9:",
            "elif sm >= 9:",
            "elif sm >= 9:",
        ),
    ],
    # The Blackwell HSTU Triton kernel only accepts head_dim ∈ {64, 128},
    # but the benchmark default is 4 heads × kv_channels=256. Re-pivot to
    # 8 heads × 128 = same 1024 hidden size, head_dim valid for sm_10.
    "generate_gin_config.py": [
        (
            "NetworkArgs.num_attention_heads = 4",
            "NetworkArgs.num_attention_heads = 8",
            "NetworkArgs.num_attention_heads = 8",
        ),
        (
            "NetworkArgs.kv_channels = 256",
            "NetworkArgs.kv_channels = 128",
            "NetworkArgs.kv_channels = 128",
        ),
    ],
}

# Global rules: applied to every .py file under the search roots.
# Use sparingly — they make the patcher quadratic-ish in disk I/O.
GLOBAL_RULES = [
    (
        "fbgemm_gpu.experimental.hstu.hstu_blackwell",
        "hstu.hstu_blackwell",
        "hstu.hstu_blackwell",
    ),
]

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
        if old not in s:
            continue
        # If marker == new, replacement is idempotent (replace won't match
        # `new` content after the first pass).
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
patched_paths: set = set()

for r in roots:
    p_root = pathlib.Path(r)
    if not p_root.exists():
        continue
    # Per-basename rules first (cheaper, targeted).
    for basename, rules in PER_FILE_RULES.items():
        for f in p_root.rglob(basename):
            n = patch_file(f, rules)
            if n:
                patched_paths.add(str(f))
                total_edits += n
    # Path-rewrite sweep, restricted to .py files whose path contains an
    # `hstu`-flavoured directory. This skips the bulk of dist-packages
    # (~50k unrelated .py files) while still catching:
    #   - hstu/ and hstu/hstu_blackwell/ in the installed package
    #   - third_party/FBGEMM/fbgemm_gpu/experimental/hstu/{,hstu/,src/hstu_blackwell/}
    # Exclude this script itself: it embeds the OLD pattern as a literal in
    # its rule list and would mangle itself otherwise.
    self_basename = pathlib.Path(__file__).name
    if GLOBAL_RULES:
        seen_in_sweep: set = set()
        for pattern in ("**/hstu/**/*.py", "**/hstu_*/**/*.py"):
            for f in p_root.glob(pattern):
                if f.name == self_basename:
                    continue
                key = str(f)
                if key in seen_in_sweep:
                    continue
                seen_in_sweep.add(key)
                n = patch_file(f, GLOBAL_RULES)
                if n:
                    patched_paths.add(key)
                    total_edits += n

print(
    f"[blackwell-patch] patched {total_edits} edit(s) "
    f"across {len(patched_paths)} file(s)"
)
