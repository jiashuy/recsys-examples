# User guide

## Working directory

`hstu/`

All commands below are run from here, and this is the PyTorch package to
optimize.

## How to start

```bash
python3 train.py \
  --dataset ml-1m --data-dir data \
  --seq-len 500 \
  --batch-size 128 \
  --embedding-dim 256 \
  --num-layers 16 \
  --num-heads 8 \
  --hidden-dim 32 \
  --attention-dim 32 \
  --lr 1e-3 \
  --dropout 0.2 \
  --steps 500
```

This is the scaled Talos performance workload. On the 5K Pro reference
host, the unoptimized path takes about 617 ms per end-to-end training
iteration and reserves about 53 GiB of GPU memory. The smaller upstream-aligned
MovieLens configuration remains documented in `README.md`.

## How to stop

Wait for the training process to finish, or use `ps` command to find pid + `kill` to terminate the process

## Focus guidance — Kernel Fusion and high performance attention custom op

**Goal: 3x lower the end-to-end latency, compared to the baseline.**

Profile-first, evidence-driven: The GPU utilization for this model is already quite high, so you'll need to look into kernel fusion and specific patterns. This includes kernel fusion for GEMM and element-wise operators and others, the design and optimization of attention operators, such as using customized variants of FlashAttention.

## Attention

Pay attention for the following:
- For workload-driven shape variation, benchmark representative buckets rather
  than only one captured input.
- Keep model-specific feature, embedding, or request-processing logic outside a
  compiled tensor region unless its runtime contract is stable.
