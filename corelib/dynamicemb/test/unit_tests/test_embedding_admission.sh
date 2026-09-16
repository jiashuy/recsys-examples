#!/bin/bash 
set -e

# Test configurations
NUM_EMBEDDING_COLLECTIONS=2
NUM_EMBEDDINGS=10000,10000,10000,10000
MULTI_HOT_SIZES=5,5,5,5
EMBEDDING_DIM=16
NUM_GPUS=(1 8)
OPTIMIZER_TYPE=("sgd" "adam" "rowwise_adagrad")
BATCH_SIZE=32
NUM_ITERATIONS=10
THRESHOLD=4
SCORE_STRATEGY=("timestamp" "lfu" "step")

# Shrink per-rank HBM budget so total table memory exceeds it → HybridStorage +
# prefetch StorageMode DEFAULT (_generic_forward_path), without enabling cache.
GLOBAL_HBM_BUDGET_SCALE_DEFAULT=0.25

# Cache configurations
CACHING_MODES=("False" "True")
CACHE_CAPACITY_RATIO=0.3  # 30% cache capacity to trigger evictions


for num_gpus in ${NUM_GPUS[@]}; do
  for optimizer_type in ${OPTIMIZER_TYPE[@]}; do
    for score_strategy in ${SCORE_STRATEGY[@]}; do
      echo ""
      echo "----------------------------------------"
      echo "Test: Storage-Only | GPUs: $num_gpus | Optimizer: $optimizer_type"
      echo "----------------------------------------"
      torchrun \
        --nnodes 1 \
        --nproc_per_node $num_gpus \
        ./test/unit_tests/test_embedding_admission.py \
        --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
        --num-embeddings $NUM_EMBEDDINGS \
        --multi-hot-sizes $MULTI_HOT_SIZES \
        --embedding-dim $EMBEDDING_DIM \
        --optimizer-type ${optimizer_type} \
        --batch-size $BATCH_SIZE \
        --num-iterations $NUM_ITERATIONS \
        --threshold $THRESHOLD \
        --score-strategy ${score_strategy} || exit 1
    done
  done
done

for num_gpus in ${NUM_GPUS[@]}; do
  for optimizer_type in ${OPTIMIZER_TYPE[@]}; do
    for score_strategy in ${SCORE_STRATEGY[@]}; do
      echo ""
      echo "----------------------------------------"
      echo "Test: StorageMode DEFAULT (Hybrid, no cache) | GPUs: $num_gpus | Optimizer: $optimizer_type | HBM scale: $GLOBAL_HBM_BUDGET_SCALE_DEFAULT"
      echo "----------------------------------------"
      torchrun \
        --nnodes 1 \
        --nproc_per_node $num_gpus \
        ./test/unit_tests/test_embedding_admission.py \
        --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
        --num-embeddings $NUM_EMBEDDINGS \
        --multi-hot-sizes $MULTI_HOT_SIZES \
        --embedding-dim $EMBEDDING_DIM \
        --optimizer-type ${optimizer_type} \
        --batch-size $BATCH_SIZE \
        --num-iterations $NUM_ITERATIONS \
        --threshold $THRESHOLD \
        --global-hbm-budget-scale $GLOBAL_HBM_BUDGET_SCALE_DEFAULT \
        --score-strategy ${score_strategy} || exit 1
    done
  done
done

for num_gpus in ${NUM_GPUS[@]}; do
  for optimizer_type in ${OPTIMIZER_TYPE[@]}; do
    for score_strategy in ${SCORE_STRATEGY[@]}; do
      echo ""
      echo "----------------------------------------"
      echo "Test: Cache+Storage | GPUs: $num_gpus | Optimizer: $optimizer_type | Cache Ratio: $CACHE_CAPACITY_RATIO"
      echo "----------------------------------------"
      torchrun \
        --nnodes 1 \
        --nproc_per_node $num_gpus \
        ./test/unit_tests/test_embedding_admission.py \
        --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
        --num-embeddings $NUM_EMBEDDINGS \
        --multi-hot-sizes $MULTI_HOT_SIZES \
        --embedding-dim $EMBEDDING_DIM \
        --optimizer-type ${optimizer_type} \
        --batch-size $BATCH_SIZE \
        --num-iterations $NUM_ITERATIONS \
        --threshold $THRESHOLD \
        --caching \
        --cache-capacity-ratio $CACHE_CAPACITY_RATIO \
        --score-strategy ${score_strategy} || exit 1
    done
  done
done



# High-frequency test: more iterations to test frequency accumulation

HIGH_FREQ_ITERATIONS=50
for caching_mode in "without-cache" "with-cache"; do
  echo ""
  echo "----------------------------------------"
  echo "Test: High Frequency ($HIGH_FREQ_ITERATIONS iters) | Mode: $caching_mode"
  echo "----------------------------------------"
  if [ "$caching_mode" = "without-cache" ]; then
    torchrun \
      --nnodes 1 \
      --nproc_per_node 1 \
      ./test/unit_tests/test_embedding_admission.py \
      --num-embedding-collections 1 \
      --num-embeddings 5000 \
      --multi-hot-sizes 3 \
      --embedding-dim $EMBEDDING_DIM \
      --optimizer-type sgd \
      --batch-size 16 \
      --num-iterations $HIGH_FREQ_ITERATIONS \
      --threshold $THRESHOLD || exit 1
  else
    torchrun \
      --nnodes 1 \
      --nproc_per_node 1 \
      ./test/unit_tests/test_embedding_admission.py \
      --num-embedding-collections 1 \
      --num-embeddings 5000 \
      --multi-hot-sizes 3 \
      --embedding-dim $EMBEDDING_DIM \
      --optimizer-type sgd \
      --batch-size 16 \
      --num-iterations $HIGH_FREQ_ITERATIONS \
      --threshold $THRESHOLD \
      --caching \
      --cache-capacity-ratio 0.4 || exit 1
  fi
done



EVICTION_CACHE_RATIO_1=0.08  # 2% - Very small cache
EVICTION_BATCH_SIZE_1=64     # Large batch size
EVICTION_ITERATIONS_1=25     # Many iterations

for optimizer_type in "sgd" "adam"; do
  echo ""
  echo "----------------------------------------"
  echo "Test: Ultra-small Cache | Optimizer: $optimizer_type | Cache Ratio: $EVICTION_CACHE_RATIO_1"
  echo "----------------------------------------"
  torchrun \
    --nnodes 1 \
    --nproc_per_node 1 \
    ./test/unit_tests/test_embedding_admission.py \
    --num-embedding-collections 1 \
    --num-embeddings 10000 \
    --multi-hot-sizes 5 \
    --embedding-dim $EMBEDDING_DIM \
    --optimizer-type ${optimizer_type} \
    --batch-size $EVICTION_BATCH_SIZE_1 \
    --num-iterations $EVICTION_ITERATIONS_1 \
    --threshold $THRESHOLD \
    --caching \
    --cache-capacity-ratio $EVICTION_CACHE_RATIO_1 || exit 1
done


# Non-admitted embedding initialization.
#
# A threshold no key can reach rejects every key, so every value the forward
# returns must come from whichever initializer owns non-admitted rows: the
# admission strategy's when it has one, the table's otherwise. Both variants run
# against all three storage configurations, because each one reaches admission
# through a different code path (HBM-direct prefetch, generic forward, cache
# prefetch).
REJECT_ALL_THRESHOLD=1000000000
NON_ADMITTED_INIT_VALUE=0.5

for num_gpus in ${NUM_GPUS[@]}; do
  for init_owner in "strategy" "table"; do
    if [ "$init_owner" = "strategy" ]; then
      INIT_ARGS="--non-admitted-init-value $NON_ADMITTED_INIT_VALUE"
    else
      INIT_ARGS="--no-strategy-initializer"
    fi
    for storage_mode in "hbm-direct" "hybrid" "cache"; do
      case "$storage_mode" in
        hbm-direct) STORAGE_ARGS="" ;;
        hybrid) STORAGE_ARGS="--global-hbm-budget-scale $GLOBAL_HBM_BUDGET_SCALE_DEFAULT" ;;
        cache) STORAGE_ARGS="--caching --cache-capacity-ratio $CACHE_CAPACITY_RATIO" ;;
      esac
      echo ""
      echo "----------------------------------------"
      echo "Test: Reject-all init | GPUs: $num_gpus | Initializer owner: $init_owner | Storage: $storage_mode"
      echo "----------------------------------------"
      torchrun \
        --nnodes 1 \
        --nproc_per_node $num_gpus \
        ./test/unit_tests/test_embedding_admission.py \
        --num-embedding-collections $NUM_EMBEDDING_COLLECTIONS \
        --num-embeddings $NUM_EMBEDDINGS \
        --multi-hot-sizes $MULTI_HOT_SIZES \
        --embedding-dim $EMBEDDING_DIM \
        --optimizer-type sgd \
        --batch-size $BATCH_SIZE \
        --num-iterations $NUM_ITERATIONS \
        --threshold $REJECT_ALL_THRESHOLD \
        --expect-all-rejected \
        $INIT_ARGS \
        $STORAGE_ARGS || exit 1
    done
  done
done
