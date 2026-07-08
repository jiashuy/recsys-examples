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

"""JIT glue for the LruLfu (LRU_LFU strategy) eviction cubins.

- ``ensure_lex_fatbin_loaded()`` hands the default (Lex) evictor fatbin to the
  C++ loader once. The default eviction path needs only this -- no numba, and the
  custom fatbin is not read/required.
- ``register_score_function(fn, gamma)`` numba-compiles a user decay function to
  LTO-IR, links it into the custom cubin (nvJitLink, C++ side), and returns an
  integer key used to route inserts to that custom evictor. Cached per function.
"""
import hashlib
import inspect
import os
import threading

import dynamicemb_extensions as _ext

_lock = threading.Lock()
_lex_loaded = False
# score_function group key (int) -> True once registered with the C++ cache.
_registered_keys = set()

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_LEX_FATBIN = os.path.join(_PKG_DIR, "evict_lrulfu_lex.fatbin")
_CUSTOM_FATBIN = os.path.join(_PKG_DIR, "evict_lrulfu_custom.fatbin")


def _read_fatbin(path: str) -> bytes:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"LruLfu evict fatbin missing: {path}. Rebuild the extension "
            "(python setup.py build_ext --inplace) so the fatbins are produced "
            "as package_data."
        )
    with open(path, "rb") as fh:
        return fh.read()


def ensure_lex_fatbin_loaded() -> None:
    """Load the default (Lex) LruLfu evict fatbin into the C++ side (idempotent).

    This is all the default eviction path needs -- no numba, and the custom
    fatbin is not read/required unless a score_function is registered."""
    global _lex_loaded
    if _lex_loaded:
        return
    with _lock:
        if _lex_loaded:
            return
        _ext.demb_set_lex_fatbin(_read_fatbin(_LEX_FATBIN))
        _lex_loaded = True


def score_function_key(fn) -> int:
    """Stable non-zero int key for a score_function (0 is reserved for default).

    Derived from (module, qualname, source hash) so identical functions share a
    key. Truncated to a positive int64."""
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = repr(fn)
    ident = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', '')}:{src}"
    digest = hashlib.sha1(ident.encode("utf-8")).hexdigest()
    key = int(digest[:15], 16)  # 60 bits, always positive, nonzero in practice
    return key or 1


def register_score_function(fn, cc_major: int, cc_minor: int) -> int:
    """numba-compile fn -> LTO-IR, link into the custom cubin, cache under its
    key. Returns the key to pass as score_fn_key on inserts. Idempotent. The
    custom fatbin is read only here (the default path never touches it)."""
    ensure_lex_fatbin_loaded()
    key = score_function_key(fn)
    if key in _registered_keys:
        return key
    with _lock:
        if key in _registered_keys:
            return key
        try:
            from numba import cuda, types
        except ImportError as e:
            raise ImportError(
                "score_function requires numba-cuda to JIT-compile the eviction "
                "decay. Install numba-cuda, or omit score_function to use the "
                "default (frequency, older-timestamp tiebreak) evictor."
            ) from e
        # scores[0]=timestamp, scores[1]=frequency (raw uint64 shared-mem words);
        # cur_ts uint64; gamma float32 -> float64. gamma is float32 to match the
        # extern user_score_fn ABI (float gamma) in evict_comparators.cuh. cc MUST
        # be the device's (the numba default sm_50 is rejected by recent CUDA NVVM).
        ltoir, _ = cuda.compile(
            fn,
            sig=(types.CPointer(types.uint64), types.uint64, types.float32),
            device=True,
            output="ltoir",
            abi="c",
            abi_info={"abi_name": "user_score_fn"},
            cc=(cc_major, cc_minor),
        )
        _ext.demb_register_score_function(
            key, bytes(ltoir), _read_fatbin(_CUSTOM_FATBIN), cc_major, cc_minor
        )
        _registered_keys.add(key)
    return key
