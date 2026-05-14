#!/usr/bin/env python3
# Idempotent in-place patch for Blackwell (sm_10+).
#
# Why:
#   examples/hstu/ops/fused_hstu_op.py unconditionally does
#       import hstu.hstu_ops_gpu
#   which tries to register a fake impl for fbgemm::hstu_varlen_fwd_80.
#   That C++ op is not shipped for sm_10 in the current container, so
#   `register_fake` raises RuntimeError at import time and the training
#   process dies before main().
#
# What this does:
#   Wraps the offending import in an SM-version check. No-op if already
#   patched, or if the file does not contain the exact target line.

import pathlib
import sys

OLD = "import hstu.hstu_ops_gpu  # noqa: F401 – registers fake impls for torch.export"
NEW = (
    "import torch as _t\n"
    "if _t.cuda.is_available() and _t.cuda.get_device_capability() < (10, 0):\n"
    "    import hstu.hstu_ops_gpu  # noqa: F401  # registers fake impls for torch.export"
)

# Roots scanned by default; override by passing paths as argv.
DEFAULT_ROOTS = [
    "/Workspace",
    "/workspace",
    "/opt",
    "/usr/local/lib",
    "/root",
]

roots = sys.argv[1:] or DEFAULT_ROOTS
patched = 0
for r in roots:
    p_root = pathlib.Path(r)
    if not p_root.exists():
        continue
    for f in p_root.rglob("fused_hstu_op.py"):
        try:
            s = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        if OLD in s and "get_device_capability" not in s:
            f.write_text(s.replace(OLD, NEW))
            print(f"[blackwell-patch] {f}")
            patched += 1

print(f"[blackwell-patch] patched {patched} file(s)")
