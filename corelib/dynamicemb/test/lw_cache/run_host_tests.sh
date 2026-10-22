#!/usr/bin/env bash
# Build and run the host-only lw_cache tests.
#
# These compile with a plain C++17 compiler against host_shim/cuda_runtime.h,
# so they run anywhere — no CUDA toolkit, no GPU.  What they cover is the
# algorithmic core: probe sequences, wraparound, collisions, capacity limits,
# rebuild-on-evict.  What they cannot cover is concurrency, memory ordering, or
# anything that actually touches a device.
#
# Usage:  ./run_host_tests.sh [output-dir]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-${TMPDIR:-/tmp}/lw_cache_host_tests}"
mkdir -p "$out"

CXX="${CXX:-g++}"
flags=(-std=c++17 -O1 -g -Wall -Wextra -Wno-unused-parameter
       -I"$here/../../src/lw_cache" -I"$here" -I"$here/host_shim")

status=0
shopt -s nullglob
for src in "$here"/test_*.cpp; do
  name="$(basename "$src" .cpp)"
  printf '=== %s ===\n' "$name"
  "$CXX" "${flags[@]}" "$src" -o "$out/$name"
  if "$out/$name"; then
    printf '%s: PASS\n\n' "$name"
  else
    printf '%s: FAIL\n\n' "$name"
    status=1
  fi
done

exit "$status"
