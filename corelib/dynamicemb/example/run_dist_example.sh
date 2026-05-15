#!/usr/bin/env bash
set -euo pipefail

# Multi-node distributed launcher for example.py.
# Equivalent to `run_example.sh`'s second invocation:
#   torchrun --standalone --nproc_per_node=${NGPU} \
#       example.py --train --caching --prefetch_pipeline "$@"
# but reshaped for srun + pyxis on a multi-node SLURM allocation.

MASTER_PORT="${MASTER_PORT:-29500}"
# Put the master-addr coordination file in CWD (set by --container-workdir),
# which is the same shared-mount path on every node — works for both
# dlcluster (/Workspace) and lyris (/lustre).
MASTER_ADDR_FILE="$(pwd)/.master_addr.example.$SLURM_JOB_ID"

{
  if [ "${SLURM_NODEID}" = "0" ]; then
    echo "${SLURMD_NODENAME:-$(hostname)}" > "$MASTER_ADDR_FILE"
  else
    while [ ! -f "$MASTER_ADDR_FILE" ]; do sleep 2; done
  fi
  MASTER_ADDR=$(cat "$MASTER_ADDR_FILE")

  torchrun \
    --nnodes="${SLURM_NNODES:-2}" \
    --nproc-per-node="${SLURM_GPUS_PER_NODE:-4}" \
    --node-rank="$SLURM_NODEID" \
    --rdzv-backend=c10d \
    --rdzv-id="$SLURM_JOB_ID" \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    ./example.py --train --caching --prefetch_pipeline "$@"

  if [ "${SLURM_NODEID}" = "0" ]; then
    rm -f "$MASTER_ADDR_FILE"
  fi
} > "example-node${SLURM_NODEID}.log" 2>&1


# launch on dlcluster (gb200nvl72)
# srun -p gb200nvl72 -A blackwell \
#        -N 2 \
#        --gpus-per-node=4 \
#        --ntasks-per-node=1 \
#        --mpi=pmix \
#        --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
#        --container-mounts=/home/scratch.jiashuy_gpu/mount:/Workspace \
#        --container-workdir=/Workspace/cc_repo/GR-mon5-11/corelib/dynamicemb/example \
#        ./run_dist_example.sh
#
# launch on lyris (gb200, exclusive nodes — no --gpus-per-node)
# srun -p gb200 -A coreai_devtech_all \
#        -N 2 --segment=2 \
#        --ntasks-per-node=1 \
#        --mpi=pmix \
#        --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
#        --container-mounts=/lustre:/lustre \
#        --container-workdir=/lustre/fsw/coreai_devtech_all/jiashuy/GR-mon5-11/corelib/dynamicemb/example \
#        ./run_dist_example.sh
