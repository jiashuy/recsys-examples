# HSTU Example

Source choice:

- Primary source: Meta's official `meta-recsys/generative-recommenders` repository, commit `47527f520b24f0ee79a7651c280c77083dbaf25b`.
- The upstream repo contains the ICML 2024 HSTU implementation, public MovieLens/Amazon training configs, synthetic-data support, and DLRM-v3 debug training.
- This directory is a compact PyTorch-only training fixture derived from the public HSTU design, not a vendored copy of the upstream package. It avoids `fbgemm_gpu`, `torchrec`, gin, and Triton/Hammer dependencies so it can run in the talos test container.
- The default command uses synthetic data for fast smoke tests. Use `--dataset ml-1m` to train on real MovieLens 1M sequences.

Run:

```bash
cd /home/ruotongw/talos
python3 examples/hstu/train.py --steps 20
```

Download real data:

```bash
cd /home/ruotongw/talos
python3 examples/hstu/prepare_movielens.py --data-dir examples/hstu/data
```

Train with real MovieLens 1M data:

```bash
cd /home/ruotongw/talos
python3 examples/hstu/train.py \
  --dataset ml-1m --data-dir examples/hstu/data \
  --seq-len 200 \
  --batch-size 128 \
  --embedding-dim 50 \
  --num-layers 8 \
  --num-heads 2 \
  --hidden-dim 25 \
  --attention-dim 25 \
  --lr 1e-3 \
  --dropout 0.2 \
  --steps 500
```

Scaled Talos performance workload (about 617 ms per unoptimized
end-to-end training iteration on the 5K Pro reference host):

```bash
cd /home/ruotongw/talos
python3 examples/hstu/train.py \
  --dataset ml-1m --data-dir examples/hstu/data \
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

The scaled workload uses the public ML-3B-large HSTU shape for sequence
length, width, depth, and attention, while retaining the local MovieLens 1M
dataset and its vocabulary. It is a performance workload, not an ML-3B
training reproduction.

The loader reads `ratings.dat`, keeps interactions with `rating >= --min-rating` (default `4.0`), sorts each user's sequence by timestamp, maps movie ids to dense item ids, and builds next-item training windows. Padding targets use item id `0` and are ignored by the loss.

Useful smoke-test variants:

```bash
python3 examples/hstu/train.py --steps 5 --batch-size 16 --seq-len 64 --embedding-dim 64 --num-layers 2
python3 examples/hstu/train.py --steps 20 --batch-size 128 --seq-len 128
python3 examples/hstu/train.py --dataset ml-1m --data-dir examples/hstu/data --steps 5 --batch-size 16 --seq-len 64 --embedding-dim 64 --num-layers 2
```

Upstream full reproduction, if the environment has the extra dependencies:

```bash
git clone https://github.com/meta-recsys/generative-recommenders.git
cd generative-recommenders
pip3 install -r requirements.txt
mkdir -p tmp/ && python3 preprocess_public_data.py
CUDA_VISIBLE_DEVICES=0 python3 main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
```
