set -e
FWD_BWD_TEST_FILES=(
    "test/unit_tests/table_operation/test_table_operation.sh"
    "test/unit_tests/retain_evicted_keys/test_retain_evicted_keys.sh"
    "test/unit_tests/test_lfu_scores.sh"
    "test/test_batched_dynamic_embedding_tables_v2.py"
    "test/test_unique_op.py"
    "test/unit_tests/test_sequence_embedding.sh"
    "test/unit_tests/test_pooled_embedding.sh"
    "test/unit_tests/test_weighted_pooled_embedding_v2.py"
    "test/unit_tests/test_twin_module.sh"
    "test/unit_tests/test_alignment.sh"
    "test/unit_tests/test_hash_roundrobin_kuairand.py"
    "test/unit_tests/test_vmm_tensor.py"
    "test/unit_tests/optimizer/test_padded_buffer_optimizer.py"
    "test/unit_tests/optimizer/test_ftrl_optimizer.py"
    # Added by #495 and never registered, so its CUDA checks on the packed
    # fp32 Adagrad accumulator -- odd dims, checkpoint conversion, state
    # precision -- have not run in CI since.
    "test/unit_tests/optimizer/test_rowwise_adagrad_state_precision.py"
    "test/unit_tests/test_planner_plan_cost.py"
    "test/unit_tests/test_planner_placement.py"
    "test/unit_tests/test_table_row_wise.sh"
    # The components of the storage and the checkpoint, the chained index, and the
    # storage below the tables. They run alone, on one GPU.
    "test/unit_tests/components/test_row_exchange.py"
    "test/unit_tests/components/test_mixed_dim_buffers.py"
    "test/unit_tests/components/test_score_rules.py"
    "test/unit_tests/components/test_eviction_sink.py"
    "test/unit_tests/components/test_row_chunk.py"
    "test/unit_tests/components/test_row_allocator.py"
    "test/unit_tests/components/test_shard_info.py"
    "test/unit_tests/components/test_cache_core.py"
    "test/unit_tests/chained_table/test_chained_table.py"
    "test/unit_tests/chained_table/test_chained_prefetch.py"
    "test/unit_tests/test_ps_tier.py"
)

LOAD_DUMP_TEST_FILES=(
    "test/unit_tests/admission/test_embedding_admission.sh"
    "test/unit_tests/test_embedding_dump_load.sh"
    "test/unit_tests/incremental_dump/test_incremental_dump.sh"
    "test/unit_tests/chained_table/test_chained_incremental.py"
    # Registered in both groups: it covers forward and checkpoints, and runs
    # only the half named by the group it is passed.
    "test/unit_tests/test_table_row_wise.sh"
)

case "$1" in
    fwd_bwd)
        TEST_FILES=("${FWD_BWD_TEST_FILES[@]}")
        ;;
    load_dump)
        TEST_FILES=("${LOAD_DUMP_TEST_FILES[@]}")
        ;;
    *)
        TEST_FILES=("${FWD_BWD_TEST_FILES[@]}" "${LOAD_DUMP_TEST_FILES[@]}")
        ;;
esac

# Run each test file using the appropriate command
for TEST_FILE in "${TEST_FILES[@]}"; do
    echo "Running tests in $TEST_FILE"
    if [[ "$TEST_FILE" == *.py ]]; then
        # Run Python test files with pytest
        pytest -svv "$TEST_FILE"
        # Check if the test failed
        if [ $? -ne 0 ]; then
            echo "ERROR: Test failed in $TEST_FILE"
            exit 1
        fi
    elif [[ "$TEST_FILE" == *.sh ]]; then
        # Run shell scripts with bash. The group travels in the environment
        # rather than as an argument, so that a script spanning both groups
        # can run only the half being asked for without the others seeing an
        # argument they did not expect -- a script that
        # forwards "$@" to pytest, which would read "fwd_bwd" as a path to
        # collect and fail.
        DYNAMICEMB_TEST_GROUP="$1" bash "$TEST_FILE"
        # Check if the test failed
        if [ $? -ne 0 ]; then
            echo "ERROR: Test failed in $TEST_FILE"
            exit 1
        fi
    else
        echo "Unknown test file type: $TEST_FILE"
    fi
done
