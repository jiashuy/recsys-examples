#!/bin/bash
cd /home/scratch.jiashuy_gpu/mount/cc_repo/GR-mon5-11/examples/hstu

./training/benchmark/scripts/submit_all_experiments_slurm.sh \
    --exp-file=training/benchmark/experiments_exp2_only.txt \
    --partition=gb200nvl72 \
    --account=blackwell \
    --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
    --nodes=18 \
    --ranks-per-node=4 \
    --time=01:00:00 \
    --wait-and-analyze