# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness and latency client for the export-aligned Python backend."""

import argparse
import time

import gin
import numpy as np
import torch
import tritonclient.http as httpclient
from commons.datasets import get_data_loader
from commons.datasets.hstu_sequence_dataset import get_dataset
from commons.utils.stringify import stringify_dict
from modules.metrics import get_multi_event_metric_module
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from tritonclient.utils import np_to_triton_dtype
from utils import DatasetArgs, RankingArgs


def _strip_padding_batch(batch, unpadded_batch_size):
    batch.batch_size = unpadded_batch_size
    kjt_dict = batch.features.to_dict()
    for key in kjt_dict:
        kjt_dict[key] = JaggedTensor.from_dense_lengths(
            kjt_dict[key].to_padded_dense()[:unpadded_batch_size],
            kjt_dict[key].lengths()[:unpadded_batch_size].long(),
        )
    batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
    batch.num_candidates = batch.num_candidates[:unpadded_batch_size]
    batch.actual_batch_size = unpadded_batch_size
    return batch


def _infer_batch(client, model_name, request_id, batch):
    user_ids = batch.features.to_dict()["user_id"].values()
    if int(user_ids.numel()) != int(batch.batch_size):
        batch = _strip_padding_batch(batch, int(user_ids.numel()))

    arrays = [
        batch.features.values().detach().cpu().numpy(),
        batch.features.lengths().detach().cpu().numpy(),
        batch.num_candidates.detach().cpu().numpy(),
    ]
    inputs = []
    for input_name, array in zip(("INPUT__0", "INPUT__1", "INPUT__2"), arrays):
        array = np.asarray(array, dtype=np.int64)
        infer_input = httpclient.InferInput(
            input_name, array.shape, np_to_triton_dtype(array.dtype)
        )
        infer_input.set_data_from_numpy(array)
        inputs.append(infer_input)

    start_time = time.perf_counter()
    response = client.infer(
        model_name,
        inputs,
        request_id=str(request_id),
        outputs=[httpclient.InferRequestedOutput("OUTPUT__0")],
    )
    return batch, response, time.perf_counter() - start_time


def _update_metrics(eval_module, batch, response):
    logits = response.as_numpy("OUTPUT__0")
    if logits is None or logits.ndim != 2 or logits.shape[1] != 8:
        raise ValueError(
            f"Unexpected OUTPUT__0 shape: {getattr(logits, 'shape', None)}"
        )
    eval_module(
        torch.from_numpy(logits).to(
            dtype=torch.bfloat16, device=torch.cuda.current_device()
        ),
        batch.labels.values().cuda(),
    )


def _logical_batch_size(batch):
    return int(batch.features.to_dict()["user_id"].values().numel())


def run(args):
    gin.parse_config_file(args.gin_config_file)
    dataset_args = DatasetArgs()
    if dataset_args.dataset_name != "kuairand-1k":
        raise ValueError(f"Unsupported dataset: {dataset_args.dataset_name}")
    if args.batch_size < 1 or args.batch_size > 8:
        raise ValueError("--batch_size must be in [1, 8]")

    ranking_args = RankingArgs()
    train_dataset, eval_dataset = get_dataset(
        dataset_name=dataset_args.dataset_name,
        dataset_path=dataset_args.dataset_path,
        max_history_seqlen=dataset_args.max_history_seqlen,
        max_num_candidates=dataset_args.max_num_candidates,
        num_tasks=ranking_args.num_tasks,
        batch_size=args.batch_size,
        rank=0,
        world_size=1,
        shuffle=False,
        random_seed=0,
        eval_batch_size=args.batch_size,
        load_candidate_action=True,
    )
    selected_dataset = train_dataset if args.train_dataset else eval_dataset
    batches = list(get_data_loader(dataset=selected_dataset))
    if not batches:
        raise RuntimeError("The selected dataset produced no batches")

    non_full_batch_indices = [
        index
        for index, batch in enumerate(batches)
        if _logical_batch_size(batch) != args.batch_size
    ]
    if non_full_batch_indices != [len(batches) - 1]:
        raise RuntimeError(
            "Expected the final batch to be the only non-full batch, got "
            f"indices {non_full_batch_indices} among {len(batches)} batches"
        )
    warmup_batch = batches[-1]
    warmup_batch_size = _logical_batch_size(warmup_batch)
    if warmup_batch_size < 1 or warmup_batch_size >= args.batch_size:
        raise RuntimeError(
            f"Invalid final warmup batch size {warmup_batch_size} for "
            f"requested batch size {args.batch_size}"
        )
    measured_batches = batches[:-1]

    with torch.inference_mode(), httpclient.InferenceServerClient(args.url) as client:
        _infer_batch(
            client,
            args.model_name,
            "warmup-final-non-full",
            warmup_batch,
        )
        print(
            "Warmup: sent the final non-full batch with "
            f"{warmup_batch_size} logical request(s); excluded from "
            "measured runs"
        )
        if args.post_warmup_sleep_seconds > 0:
            time.sleep(args.post_warmup_sleep_seconds)

        elapsed_runs = []
        for run_index in range(1, args.num_runs + 1):
            run_seconds = 0.0
            batch_count = 0
            logical_request_count = 0
            for batch_index, batch in enumerate(measured_batches):
                batch, response, elapsed_seconds = _infer_batch(
                    client,
                    args.model_name,
                    f"{run_index}-{batch_index}",
                    batch,
                )
                run_seconds += elapsed_seconds
                batch_count += 1
                logical_request_count += int(batch.actual_batch_size)

            if logical_request_count == 0:
                raise RuntimeError("No measured requests were sent")
            elapsed_runs.append(run_seconds)
            print(
                f"Run {run_index}: {run_seconds:.6f} seconds E2E; "
                f"{run_seconds * 1000.0 / batch_count:.3f} ms/Triton request; "
                f"{run_seconds * 1000.0 / logical_request_count:.3f} "
                f"ms/logical request "
                f"({batch_count} batches, {logical_request_count} samples)"
            )

        # Keep GPU metric kernels completely outside the timed request loops.
        # Although _infer_batch stops its timer before this work, scheduling
        # AUC kernels between requests can contend with the Triton model on the
        # same GPU and inflate the following synchronous client.infer call.
        eval_module = get_multi_event_metric_module(
            num_classes=ranking_args.prediction_head_arch[-1],
            num_tasks=ranking_args.num_tasks,
            metric_types=ranking_args.eval_metrics,
        )
        auc_batch_count = 0
        for batch_index, batch in enumerate(batches):
            batch, response, _ = _infer_batch(
                client,
                args.model_name,
                f"auc-{batch_index}",
                batch,
            )
            _update_metrics(eval_module, batch, response)
            auc_batch_count += 1
        torch.cuda.synchronize()
        print(
            f"AUC validation pass: {auc_batch_count} batches; "
            "excluded from all reported latency"
        )

    print(
        f"Average E2E over {args.num_runs} run(s): "
        f"{sum(elapsed_runs) / len(elapsed_runs):.6f} seconds"
    )
    print(
        "[eval]:\n    "
        + stringify_dict(eval_module.compute(), prefix="Metrics", sep="\n    ")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test the export-aligned HSTU Triton Python backend"
    )
    parser.add_argument("--gin_config_file", required=True)
    parser.add_argument("--url", default="localhost:8000")
    parser.add_argument("--model_name", default="hstu_export_aligned")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_runs", type=int, default=3)
    parser.add_argument("--post_warmup_sleep_seconds", type=float, default=1.0)
    parser.add_argument("--train_dataset", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
