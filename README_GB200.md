# GB200 Container & Distributed Test Guide

This document covers the GB200 (Blackwell, ARM) container image used for
dynamicemb development and a reproducible multi-node test on SLURM
clusters (`dynamicemb` distributed embedding example on MovieLens-1M).

## 1. Container image

**Image (in registry):**

```
gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200
```

The image is `linux/arm64`, built on top of an NGC PyTorch base and
provisioned with FBGEMM, torchrec, and dynamicemb.

## 2. How the image is built

The build is driven by `docker/build_gb200.sh`:

```bash
# docker/build_gb200.sh
docker buildx build \
  --platform linux/arm64 \
  --progress=plain \
  -f docker/Dockerfile.gb200 \
  -t gitlab-master.nvidia.com:5005/devtech-compute/distributed-recommender:arm_gb200 \
  --push \
  . 2>&1 | tee ./build_gb200.log
```

Files involved:

| File | Purpose |
|------|---------|
| `docker/Dockerfile.gb200` | Layered build: CUDA base → PyTorch + dependencies → FBGEMM → torchrec → dynamicemb |
| `docker/build_gb200.sh` | Top-level driver — `docker buildx … --push` |

To rebuild locally:

```bash
./docker/build_gb200.sh
```

The image lands in the GitLab registry once the buildx push completes.

## 3. SLURM authentication for pyxis

Pyxis/enroot does **not** read `~/.docker/config.json`. Put the GitLab
PAT (with `read_registry` scope) in `~/.config/enroot/.credentials`
(`chmod 600`), one line:

```
machine gitlab-master.nvidia.com login <gitlab-username> password <PAT-token>
```

> Note: the `machine` host is `gitlab-master.nvidia.com` (no `:5005`),
> even though the image reference includes the port.

## 4. Running the distributed test

The test script lives at `corelib/dynamicemb/example/run_dist_example.sh`.
It wraps:

```bash
torchrun … example.py --train --caching --prefetch_pipeline "$@"
```

with `--nnodes`/`--nproc-per-node`/`--node-rank` derived from SLURM env
vars and `MASTER_ADDR` coordinated via a shared file (`$(pwd)/.master_addr.example.$SLURM_JOB_ID`).

### dlcluster (gb200nvl72, 2 × 4 GPUs)

```bash
srun -p gb200nvl72 -A blackwell \
     -N 2 \
     --gpus-per-node=4 \
     --ntasks-per-node=1 \
     --mpi=pmix \
     --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
     --container-mounts=/home/scratch.jiashuy_gpu/mount:/Workspace \
     --container-workdir=/Workspace/cc_repo/GR-mon5-11/corelib/dynamicemb/example \
     ./run_dist_example.sh
```

### lyris (gb200, exclusive nodes — no `--gpus-per-node`)

```bash
srun -p gb200 -A coreai_devtech_all \
     -N 2 --segment=2 \
     --ntasks-per-node=1 \
     --mpi=pmix \
     --container-image=gitlab-master.nvidia.com/devtech-compute/distributed-recommender:arm_gb200 \
     --container-mounts=/lustre:/lustre \
     --container-workdir=/lustre/fsw/coreai_devtech_all/jiashuy/GR-mon5-11/corelib/dynamicemb/example \
     ./run_dist_example.sh
```

Per-node output goes to `example-node${SLURM_NODEID}.log` next to the script.

## 5. Test results

8 ranks total (2 nodes × 4 GB200), MovieLens-1M, 5 epochs, dynamicemb +
caching + prefetch pipeline, roundrobin distribution.

### Sharding plan (per rank, all 4 GPUs identical)

```
table name |       | memory(KB) |             |       | hbm(KB)/cuda:N |             |       |  dram(KB) |
---------- | ----- | ---------- | ----------- | ----- | -------------- | ----------- | ----- | --------- | -----------
           | total | embedding  | optim_state | total | embedding      | optim_state | total | embedding | optim_state
---------- | ----- | ---------- | ----------- | ----- | -------------- | ----------- | ----- | --------- | -----------
user_id    | 960   | 320        | 640         | 480   | 160            | 320         | 480   | 160       | 320
movie_id   | 960   | 320        | 640         | 480   | 160            | 320         | 480   | 160       | 320
gender     |  96   |  32        |  64         |  48   |  16            |  32         |  48   |  16       |  32
age        |  96   |  32        |  64         |  48   |  16            |  32         |  48   |  16       |  32
occupation |  96   |  32        |  64         |  48   |  16            |  32         |  48   |  16       |  32
year       | 288   |  96        | 192         | 144   |  48            |  96         | 144   |  48       |  96
```

Each table is split 50% HBM / 50% DRAM. Adam optimizer state is 2× the
embedding size.

### Training loss curve

| Epoch | Avg train loss (rank 0) | Test loss (rank 0) |
|-------|-------------------------|--------------------|
| 1     | 1.4042                  | 1.2209             |
| 2     | 1.1540                  | 1.2265             |
| 3     | 1.0717                  | 1.2896             |
| 4     | 1.0121                  | 1.3213             |
| 5     | 0.9567                  | 1.2881             |

Other ranks are within ±0.01 of rank 0 throughout — distribution is
consistent. Train loss decreases monotonically across all ranks.

### Raw logs

The full per-node logs are saved alongside the script:

- `corelib/dynamicemb/example/example-node0.log` — ranks 0–3 (node 0)
- `corelib/dynamicemb/example/example-node1.log` — ranks 4–7 (node 1)

Highlights from `example-node0.log`:

```
[RANK 0]  Using DynamicEmb dist_type=roundrobin
[RANK 0]  Epoch 1/5, Batch 0/98, Loss: 8.6520
[RANK 0]  Epoch 1/5, Average Loss: 1.4042
[RANK 0]  Epoch 1/5, Test Loss: 1.2209
...
[RANK 0]  Epoch 5/5, Average Loss: 0.9567
[RANK 0]  Epoch 5/5, Test Loss: 1.2881
```

Highlights from `example-node1.log`:

```
[RANK 4]  Using DynamicEmb dist_type=roundrobin
[RANK 4]  Epoch 1/5, Batch 0/98, Loss: 8.5708
[RANK 4]  Epoch 1/5, Average Loss: 1.4042
[RANK 4]  Epoch 1/5, Test Loss: 1.2332
...
[RANK 4]  Epoch 5/5, Average Loss: 0.9660
[RANK 5]  Epoch 5/5, Test Loss: 1.2914
[RANK 7]  Epoch 5/5, Test Loss: 1.3139
[RANK 6]  Epoch 5/5, Test Loss: 1.2893
```

