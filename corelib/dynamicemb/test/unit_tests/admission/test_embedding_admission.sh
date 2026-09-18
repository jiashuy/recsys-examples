#!/bin/bash
# Everything about admission. The strategies that decide on their own, then the
# end-to-end suite, which lives in test_embedding_admission.py: it names its
# cases and the process counts each is worth running at, so the only thing left
# to decide out here is that count, because torchrun decides it.
set -e

ADMISSION=./test/unit_tests/admission
TEST=$ADMISSION/test_embedding_admission.py

# Cheap and needs one GPU, so it runs first and fails before the 70 below.
pytest -svv $ADMISSION/test_probabilistic_admission.py || exit 1

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
