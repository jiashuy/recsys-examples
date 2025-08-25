#!/bin/bash 
set -e

CUDA_VISIBLE_DEVICES=0 torchrun \
  --nnodes 1 \
  --nproc_per_node 1 \
  -m pytest -s \
  test/unit_tests/test_dynamicemb_table_dump_load.py::test_dynamic_table_load_dump[16-100-training-training-adam-sgd-timestamp-uint64-float32-int64] 

  # test/unit_tests/test_dynamicemb_table_dump_load.py
