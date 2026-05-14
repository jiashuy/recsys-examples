#!/usr/bin/env bash
set -euo pipefail



MASTER_PORT=29500
MASTER_ADDR_FILE=/Workspace/.master_addr.$SLURM_JOB_ID

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
    ./test/unit_tests/test_sequence_embedding_fw.py \
      --print_sharding_plan \
      --optimizer_type adam \
      --use_index_dedup True \
      --batch_size 1024 \
      --num_embeddings_per_feature=8388608,4194304,524288,1048576

  if [ "${SLURM_NODEID}" = "0" ]; then
    rm -f "$MASTER_ADDR_FILE"
  fi
} > "utest-node${SLURM_NODEID}.log" 2>&1


#launch above task on dlcluster
# srun -p gb200nvl72 -A blackwell \
#        -N 2 \
#        --gpus-per-node=4 \
#        --ntasks-per-node=1 \
#        --mpi=pmix \
#        --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
#        --container-mounts=/home/scratch.jiashuy_gpu/mount:/Workspace \
#        --container-workdir=/Workspace/cc_repo/GR-mon5-11/corelib/dynamicemb \
#        ./run_dist_test.sh
