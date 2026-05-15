#!/bin/bash

#dlcluster
cd /home/scratch.jiashuy_gpu/mount/cc_repo/GR-mon5-11/examples/hstu

./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments_exp2_only.txt \
    --partition=gb200nvl72 \
    --account=blackwell \
    --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
    --nodes=18 \
    --segment=18 \
    --ranks-per-node=4 \
    --time=01:00:00 \
    --wait-and-analyze


./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments_exp2_only.txt \
    --partition=gb200nvl72 \
    --account=blackwell \
    --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
    --nodes=2 \
    --segment=2 \
    --ranks-per-node=4 \
    --time=01:00:00 \
    --wait-and-analyze


# lyris
#srun -A coreai_devtech_all  -J coreai_devtech_all-8gpu  -p gb200 -N 2 --segment=2 --pty bash
./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments_exp2_only.txt \
    --partition=gb200 \
    --account=coreai_devtech_all \
    --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
    --nodes=18 \
    --segment=18 \
    --ranks-per-node=4 \
    --time=01:00:00 \
    --wait-and-analyze

./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments_exp2_only.txt \
    --partition=gb200 \
    --account=coreai_devtech_all \
    --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
    --nodes=2 \
    --segment=2 \
    --ranks-per-node=4 \
    --time=01:00:00 \
    --wait-and-analyze


HSTU_ROOT=/lustre/fsw/coreai_devtech_all/jiashuy/GR-mon5-11/examples/hstu

srun -A coreai_devtech_all -p gb200 \
    --nodes=1 --segment=1 \
    --gpus-per-node=4 \
    --ntasks=1 \
    --time=01:00:00 \
    --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
    --container-mounts=/lustre:/lustre \
    --container-workdir=${HSTU_ROOT} \
    --pty bash -c "
python ${HSTU_ROOT}/training/benchmark/scripts/patch_hstu_blackwell.py
echo '===== patch done, entering shell ====='
export PYTHONPATH=${HSTU_ROOT}:${HSTU_ROOT}/..:\${PYTHONPATH}
export PYTHONUNBUFFERED=1
export DISABLE_RICH=1
export RANK=0
export LOCAL_RANK=0
export WORLD_SIZE=1
export LOCAL_WORLD_SIZE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export MEM_DEBUG=0
export CUDA_MEM_WATCHDOG=0
export CACHE_DEBUG=0
export FILL_DYNAMICEMB_TABLES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_MODULE_LOADING=EAGER
exec bash
"