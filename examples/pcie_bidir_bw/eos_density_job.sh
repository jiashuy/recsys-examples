#!/bin/bash
#SBATCH --account=coreai_devtech_all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=00:40:00
#SBATCH --job-name=pcie_density
#SBATCH --output=%x-%j.log
# ---------------------------------------------------------------------------
# Does sorting a step's miss rows still pay at the density a step really has?
#
# The earlier gather benchmark measured 36% density, where sorted order is close
# to a sequential scan. A training step is around 1e-4. This sweeps between the
# two, sorting within a step only, and walking a large index pool so no slice's
# address translations stay warm for the next.
#
# The answer decides design doc P1.5 (sort the miss batch before transferring).
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
nvcc -O3 -std=c++17 -gencode arch=compute_${ARCH},code=sm_${ARCH} \
     -o pcie_density_sweep pcie_density_sweep.cu 2>&1 | head -20
[ -x ./pcie_density_sweep ] || { echo "COMPILE FAILED"; exit 1; }

for GB in 16 64; do
  for DIM in 384 128; do
    echo
    echo "################ table=${GB} GiB  dim=${DIM} ($((DIM*4)) B/row) ################"
    ./pcie_density_sweep $GB $DIM 4
  done
done
'
RC=$?
echo
echo "=== srun exit: $RC ==="
date
exit $RC
