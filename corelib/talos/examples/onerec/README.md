# OneRec Example

Source choice:

- Original paper: `OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment`, arXiv `2502.18965`.
- Most official project found: Kuaishou's `Kuaishou-OneRec/OpenOneRec`, commit `a969edcadd579a06c1966ae1db5984e02f48beff`. It contains the OneRec-Foundation framework, RecIF-Bench data tooling, and Qwen3-based pretrain/post-train pipeline.
- Best lightweight training-code reference found: `Applied-Machine-Learning-Lab/KDD2026-HRPO`, commit `cef658bf6530762e40be02d10f882fbd9c4ce01d`. It includes OneRec-family models, a `train_onerec_value.py` entry point, and KuaiRand/KuaiSim data flow.
- This directory is a compact PyTorch-only training fixture. It keeps the practical OneRec-family structure: history encoder, shift-right semantic-ID decoder, constrained-token classification objective, and auxiliary reward/LTV value heads.
- The default command uses synthetic data for fast smoke tests. Use `--dataset kuairand-pure` to train on real KuaiRand-Pure interaction logs.

Run:

```bash
cd /home/ruotongw/talos
python3 examples/onerec/train.py --steps 20
```

Download real data:

```bash
cd /home/ruotongw/talos
python3 examples/onerec/prepare_kuairand.py --data-dir examples/onerec/data
```

Train with real KuaiRand-Pure data:

```bash
cd /home/ruotongw/talos
python3 examples/onerec/train.py \
  --dataset kuairand-pure \
  --data-dir examples/onerec/data \
  --steps 50 \
  --batch-size 64 \
  --max-hist-len 64 \
  --num-classes 64 \
  --sid-depth 4
```

Scaled Talos performance workload (about 620 ms per unoptimized
end-to-end training iteration on the 5K Pro reference host):

```bash
cd /home/ruotongw/talos
python3 examples/onerec/train.py \
  --dataset kuairand-pure \
  --data-dir examples/onerec/data \
  --steps 50 \
  --batch-size 512 \
  --max-hist-len 256 \
  --num-classes 32 \
  --sid-depth 4 \
  --dim 512 \
  --num-heads 8 \
  --encoder-layers 4 \
  --decoder-layers 4 \
  --value-layers 2 \
  --dropout 0.1 \
  --lr 1e-4 \
  --reward-weight 0.5 \
  --ltv-weight 0.5 \
  --gamma 0.99
```

The scaled workload keeps the source-aligned SID shape (`sid_depth=4`,
`num_classes=32`) and scales only the history length and dense model capacity.

The loader reads the standard KuaiRand-Pure logs, sorts each user's videos by timestamp, builds history-to-next-video training examples, and uses the real feedback columns (`is_click`, `long_view`, likes/follows/comments/forwards, watch ratio, hate) to construct reward and LTV labels. It maps real video ids to deterministic hierarchical SID tokens. For paper-level reproduction, replace this deterministic SID mapping with a learned tokenizer such as the OpenOneRec residual-kmeans tokenizer.

Useful smoke-test variants:

```bash
python3 examples/onerec/train.py --steps 5 --batch-size 16 --max-hist-len 32 --dim 64 --encoder-layers 1 --decoder-layers 1
python3 examples/onerec/train.py --steps 20 --batch-size 128 --max-hist-len 64
python3 examples/onerec/train.py --dataset kuairand-pure --data-dir examples/onerec/data --steps 5 --batch-size 16 --max-hist-len 32 --dim 64 --encoder-layers 1 --decoder-layers 1 --value-layers 1
```

Upstream full-data training reference:

```bash
git clone https://github.com/Applied-Machine-Learning-Lab/KDD2026-HRPO.git
cd KDD2026-HRPO
export PROJECT_ROOT="$PWD"
export DATA_ROOT="$PROJECT_ROOT/dataset/kuairand/kuairand-Pure/data"
# Put the KuaiSim-format KuaiRand-Pure CSV files under $DATA_ROOT, then:
bash "$PROJECT_ROOT/code/train_onerec_value.sh"
```

OpenOneRec foundation-model pipeline reference:

```bash
git clone https://github.com/Kuaishou-OneRec/OpenOneRec.git
cd OpenOneRec/pretrain
bash examples/pretrain_stg1.sh
```
