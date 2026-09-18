#!/bin/bash
# The suite itself lives in test_embedding_admission.py, which names its cases
# and the process counts each is worth running at. Only the process count has
# to be decided out here, because torchrun decides it.
set -e

TEST=./test/unit_tests/admission/test_embedding_admission.py

for num_gpus in 1 8; do
  for case_name in $(python3 "$TEST" --list-cases --num-gpus "$num_gpus"); do
    echo ""
    echo "----------------------------------------"
    echo "Test: $case_name | GPUs: $num_gpus"
    echo "----------------------------------------"
    torchrun --nnodes 1 --nproc_per_node "$num_gpus" "$TEST" --case "$case_name" \
      || exit 1
  done
done
