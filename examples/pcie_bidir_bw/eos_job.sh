#!/bin/bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --job-name=pcie_bidir
#SBATCH --output=%x-%j.log
# ---------------------------------------------------------------------------
# Measure PCIe behaviour for the two open questions in
# docs/lightweight_cache_design.md:
#
#   §12.1  Is the link full duplex in practice? The design fuses the victim
#          writeback and the miss load into one warp so both directions are in
#          flight at once. In steady state their volumes are equal — every miss
#          causes exactly one eviction — which is the ideal input for a
#          full-duplex link and worthless if it is not one.
#
#   §12.2  Does sorting the victim rows by backing row index pay? Eviction
#          produces scattered rows, so the D2H side is a random scatter.
#
# Three programs, in increasing fidelity to what the cache actually does:
#   pcie_bidir_bw       copy engines, contiguous — the ceiling
#   pcie_bidir_kernel   kernel, zero-copy, contiguous
#   pcie_bidir_gather   kernel, zero-copy, one warp per row, random rows  <-- us
#
# Row widths are the ones that matter: 384 floats is dim 128 with Adam's two
# states, which is the worked example throughout the design doc.
# ---------------------------------------------------------------------------
set -uo pipefail

IMG='gitlab-master.nvidia.com#devtech-compute/distributed-recommender:devel_latest'

echo "=== node: $(hostname) ==="
date

srun --ntasks=1 --container-image="$IMG" \
     --container-mounts="$HOME/lw_cache_test:/work" \
     bash -c '
set -uo pipefail
cd /work/pcie_bidir_bw || exit 1

ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ".")
echo "=== device ==="
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv | head -2

echo
echo "=== build ==="
for f in pcie_bidir_bw pcie_bidir_kernel pcie_bidir_gather; do
  nvcc -O3 -std=c++17 -gencode arch=compute_${ARCH},code=sm_${ARCH} -o $f $f.cu \
    2>&1 | head -20
  [ -x ./$f ] && echo "  $f OK" || echo "  $f FAILED"
done

echo
echo "########## 1. copy engines, contiguous (the ceiling) ##########"
./pcie_bidir_bw 2>&1 | tail -30

echo
echo "########## 2. kernel, zero-copy, contiguous ##########"
./pcie_bidir_kernel 2>&1 | tail -30

echo
echo "########## 3. kernel, zero-copy, gather/scatter (what the cache does) ##########"
# args: table_GB_each entries_M dim iters do_sort
for DIM in 384 128 32; do
  for SORT in 0 1; do
    echo
    echo "--- dim=$DIM ($((DIM*4)) B/row)  sorted=$SORT ---"
    ./pcie_bidir_gather 16 4 $DIM 10 $SORT 2>&1 | tail -12
  done
done
'
RC=$?
echo
echo "=== srun exit: $RC ==="
date
exit $RC
