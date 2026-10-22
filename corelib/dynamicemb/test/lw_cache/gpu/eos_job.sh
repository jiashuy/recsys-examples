#!/bin/bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --job-name=lw_cache
#SBATCH --output=%x-%j.log
# ---------------------------------------------------------------------------
# Compile and run the lw_cache tests on EOS.
#
# Batch rather than an interactive allocation: salloc needs a client to stay
# connected, and the result of this is a log file either way.
#
# Two layers, deliberately separate:
#   the standalone test builds in seconds with plain nvcc, so a failure there
#   belongs to the cache; the extension build drags in torch and pybind, so a
#   failure there belongs to the binding.
# ---------------------------------------------------------------------------
set -uo pipefail

IMG='gitlab-master.nvidia.com#devtech-compute/distributed-recommender:devel_latest'

echo "=== node: $(hostname) ==="
date

srun --ntasks=1 --container-image="$IMG" \
     --container-mounts="$HOME/lw_cache_test:/work" \
     bash -c '
set -uo pipefail
cd /work/dynamicemb || exit 1
FAILED=0

echo "=== toolchain ==="
nvcc --version | tail -2
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader | head -1
ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")
python -c "import torch; print(\"torch\", torch.__version__)" 2>/dev/null || echo "no torch"

echo
echo "=== 1. host tests ==="
( cd test/lw_cache && ./run_host_tests.sh 2>&1 | grep -E "PASS|FAIL|error:" ) || FAILED=1

echo
echo "=== 2. standalone GPU test: compile ==="
cd test/lw_cache/gpu
rm -f test_gpu
nvcc -O2 -std=c++17 -arch=sm_${ARCH} \
     --expt-relaxed-constexpr --expt-extended-lambda \
     -I../../../src/lw_cache -o test_gpu test_gpu.cu 2>&1 | head -50
if [ ! -x ./test_gpu ]; then echo "COMPILE FAILED"; exit 1; fi
echo "compile OK"

echo
echo "=== 3. standalone GPU test: run ==="
./test_gpu || FAILED=1

echo
echo "=== 4. racecheck ==="
# The point of being on a device: the host shim serialises atomics, so nothing
# before this can see a race.
compute-sanitizer --tool racecheck --racecheck-report analysis ./test_gpu 2>&1 \
  | grep -E "RACECHECK SUMMARY|hazard|FAIL|passed" | head -20

echo
echo "=== 5. memcheck ==="
compute-sanitizer --tool memcheck ./test_gpu 2>&1 \
  | grep -E "ERROR SUMMARY|FAIL|passed" | head -20

echo
echo "=== 6. extension build (binding.cu through nvcc + torch) ==="
cd /work/dynamicemb
# One arch only: the four-way fat binary is minutes of nvcc for no benefit here.
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=32
# setup.py rather than pip: this image has a preinstalled package with broken
# metadata, and pips resolver aborts on it before reaching our build. Nothing
# to do with this code, so go around it.
python setup.py build_ext --inplace > /tmp/build.log 2>&1
BUILD_RC=$?
export PYTHONPATH=/work/dynamicemb:${PYTHONPATH:-}
if [ $BUILD_RC -ne 0 ]; then
  echo "BUILD FAILED (rc=$BUILD_RC), last 60 lines:"
  tail -60 /tmp/build.log
  FAILED=1
else
  echo "build OK"
  grep -iE "lw_cache.*(warning|error)" /tmp/build.log | head -10
  ls -la dynamicemb_extensions*.so 2>/dev/null | head -2
fi

if [ $BUILD_RC -eq 0 ]; then
  echo
  echo "=== 7. bindings import ==="
  python -c "
from dynamicemb_extensions import LwCache, LwStubBacking, LwMultiCache, LwMultiStubBacking
print(\"imported:\", LwCache, LwMultiCache)
c = LwCache(capacity=64, row_elems=4, record_metrics=True)
print(\"capacity\", c.capacity, \"buckets\", c.num_buckets, \"meta_bytes\", c.metadata_bytes)
" || FAILED=1

  echo
  echo "=== 8. python differential tests ==="
  python test/lw_cache/test_differential.py --seeds 10 --steps 200 || FAILED=1
  echo
  python test/lw_cache/test_multi_table.py || FAILED=1
fi

echo
echo "=== overall: $FAILED (0 = all good) ==="
exit $FAILED
'
RC=$?
echo
echo "=== srun exit: $RC ==="
date
exit $RC
