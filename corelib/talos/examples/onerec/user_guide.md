# User guide

## Working directory

`onerec/`

All commands below are run from here, and this is the PyTorch package to
optimize.

## How to start

```bash
python3 train.py \
  --dataset kuairand-pure \
  --data-dir data \
  --batch-size 512 \
  --max-hist-len 256 \
  --sid-depth 4 \
  --num-classes 32 \
  --dim 512 \
  --num-heads 8 \
  --encoder-layers 4 \
  --decoder-layers 4 \
  --value-layers 2 \
  --dropout 0.1 \
  --lr 1e-4 \
  --reward-weight 0.5 \
  --ltv-weight 0.5 \
  --gamma 0.99 \
  --steps 50
```

This is the scaled Talos performance workload. It preserves the original
SID task shape (`sid_depth=4`, `num_classes=32`) while scaling history length,
model width, and depth. On the 5K Pro reference host, the unoptimized path
takes about 620 ms per end-to-end training iteration and reserves about 28 GiB
of GPU memory. The smaller source-aligned configuration remains documented in
`README.md`.

## How to stop

Wait for the training process to finish, or use `ps` command to find pid + `kill` to terminate the process

## Focus guidance — Kernel Fusion and high performance attention custom op

**Goal: 3x lower the end-to-end latency, compared to the baseline.**

Profile-first, evidence-driven: The GPU utilization for this model is high, but still a little room to improve. And the GPU idle gap between steps worth notice. Mainly you'll need to look into kernel fusion and specific patterns. This includes kernel fusion for GEMM and element-wise operators and others, the design and optimization of attention operators, such as using customized variants of FlashAttention. Other methods are also welcome.

## Attention

Pay attention for the following:
- For workload-driven shape variation, benchmark representative buckets rather
  than only one captured input.
- Keep model-specific feature, embedding, or request-processing logic outside a
  compiled tensor region unless its runtime contract is stable.
