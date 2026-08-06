# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_TENSORS = [
    ("INPUT__0", "values"),
    ("INPUT__1", "lengths"),
    ("INPUT__2", "num_candidates"),
    ("INPUT__3", "user_ids"),
    ("INPUT__4", "total_history_lengths"),
]
InputCase = tuple[int, list[np.ndarray]]
SUPPORTED_BATCH_SIZES = (2, 4, 8)
WARMUP_COUNT = 2
CACHE_MEASUREMENT_SET_COUNT = 3


@dataclass(frozen=True)
class InputSample:
    feature_values: tuple[np.ndarray, ...]
    feature_lengths: np.ndarray
    num_candidates: np.ndarray
    user_ids: np.ndarray
    total_history_lengths: np.ndarray


def _load_dumped_tensor(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing dumped tensor: {path}")
    module = torch.jit.load(str(path), map_location="cpu")
    tensor = module.tensor.detach().cpu().contiguous()
    return tensor.numpy()


def _make_input(httpclient, name: str, array: np.ndarray):
    if array.dtype != np.int64:
        array = array.astype(np.int64, copy=False)
    infer_input = httpclient.InferInput(name, array.shape, "INT64")
    infer_input.set_data_from_numpy(array)
    return infer_input


def _find_batch_indices(dump_dir: Path) -> list[int]:
    indices = []
    for values_path in dump_dir.glob("batch_*_values.pt"):
        batch_id = values_path.name.removeprefix("batch_").removesuffix("_values.pt")
        indices.append(int(batch_id))
    return sorted(indices)


def _load_input_cases(dump_dir: Path) -> list[InputCase]:
    input_cases = []
    for batch_index in _find_batch_indices(dump_dir):
        prefix = dump_dir / f"batch_{batch_index:06d}"
        input_cases.append(
            (
                batch_index,
                [
                    _load_dumped_tensor(Path(f"{prefix}_{suffix}.pt"))
                    for _, suffix in INPUT_TENSORS
                ],
            )
        )
    if not input_cases:
        raise FileNotFoundError(f"No dumped input cases found in {dump_dir}")
    return input_cases


def _make_inputs(httpclient, input_case: list[np.ndarray]):
    return [
        _make_input(httpclient, input_name, array)
        for (input_name, _), array in zip(INPUT_TENSORS, input_case)
    ]


def _logical_batch_size(input_case: Sequence[np.ndarray]) -> int:
    values, lengths, num_candidates, user_ids, total_history_lengths = input_case
    for name, array in (
        ("values", values),
        ("lengths", lengths),
        ("num_candidates", num_candidates),
        ("user_ids", user_ids),
        ("total_history_lengths", total_history_lengths),
    ):
        if array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {array.shape}")

    batch_size = int(user_ids.size)
    if batch_size < 1 or batch_size > max(SUPPORTED_BATCH_SIZES):
        raise ValueError(f"Unsupported logical batch size: {batch_size}")
    if num_candidates.size != batch_size:
        raise ValueError(
            "num_candidates and user_ids disagree on logical batch size: "
            f"{num_candidates.size} versus {batch_size}"
        )
    if total_history_lengths.size != batch_size:
        raise ValueError(
            "total_history_lengths and user_ids disagree on logical batch size: "
            f"{total_history_lengths.size} versus {batch_size}"
        )
    if lengths.size % batch_size != 0:
        raise ValueError(
            f"{lengths.size} feature lengths are not divisible by batch size "
            f"{batch_size}"
        )
    if np.any(lengths < 0):
        raise ValueError("Feature lengths must be non-negative")
    if int(lengths.sum()) != values.size:
        raise ValueError(
            f"Feature lengths sum to {int(lengths.sum())}, but values has "
            f"{values.size} elements"
        )
    return batch_size


def _iter_input_samples(input_cases: Sequence[InputCase]) -> Iterator[InputSample]:
    expected_num_features = None
    for _, input_case in input_cases:
        batch_size = _logical_batch_size(input_case)
        values, lengths, num_candidates, user_ids, total_history_lengths = input_case
        num_features = lengths.size // batch_size
        if expected_num_features is None:
            expected_num_features = num_features
        elif num_features != expected_num_features:
            raise ValueError(
                "Input cases disagree on feature count: "
                f"{num_features} versus {expected_num_features}"
            )

        lengths_by_feature = lengths.reshape(num_features, batch_size)
        offsets = np.empty(lengths.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, dtype=np.int64, out=offsets[1:])
        for sample_index in range(batch_size):
            feature_values = tuple(
                values[
                    offsets[feature_index * batch_size + sample_index] : offsets[
                        feature_index * batch_size + sample_index + 1
                    ]
                ]
                for feature_index in range(num_features)
            )
            yield InputSample(
                feature_values=feature_values,
                feature_lengths=lengths_by_feature[:, sample_index],
                num_candidates=num_candidates[sample_index : sample_index + 1],
                user_ids=user_ids[sample_index : sample_index + 1],
                total_history_lengths=total_history_lengths[
                    sample_index : sample_index + 1
                ],
            )


def _merge_samples(samples: Sequence[InputSample]) -> list[np.ndarray]:
    if not samples:
        raise ValueError("Cannot build an input batch from zero samples")

    num_features = len(samples[0].feature_values)
    if any(len(sample.feature_values) != num_features for sample in samples):
        raise ValueError("Input samples disagree on feature count")

    values = np.concatenate(
        [
            sample.feature_values[feature_index]
            for feature_index in range(num_features)
            for sample in samples
        ]
    )
    lengths = np.concatenate(
        [
            np.asarray(
                [
                    sample.feature_lengths[feature_index]
                    for sample in samples
                ],
                dtype=samples[0].feature_lengths.dtype,
            )
            for feature_index in range(num_features)
        ]
    )
    return [
        values,
        lengths,
        np.concatenate([sample.num_candidates for sample in samples]),
        np.concatenate([sample.user_ids for sample in samples]),
        np.concatenate([sample.total_history_lengths for sample in samples]),
    ]


def _rebatch_input_cases(
    input_cases: Sequence[InputCase], batch_size: int
) -> list[InputCase]:
    samples = list(_iter_input_samples(input_cases))
    if len(samples) % batch_size != 0:
        raise ValueError(
            f"{len(samples)} measured samples cannot form only full "
            f"batch_size={batch_size} batches. The final source batch must be "
            "the only non-full batch so it can be reserved for warmup."
        )

    return [
        (
            batch_index,
            _merge_samples(samples[start : start + batch_size]),
        )
        for batch_index, start in enumerate(range(0, len(samples), batch_size))
    ]


def _prepare_request_plan(
    input_cases: Sequence[InputCase], batch_size: int
) -> tuple[InputCase, list[InputCase]]:
    if len(input_cases) < 2:
        raise RuntimeError("Need at least two source batches to warm up and measure")

    source_batch_sizes = [
        _logical_batch_size(input_case) for _, input_case in input_cases
    ]
    measured_source_batch_sizes = source_batch_sizes[:-1]
    if len(set(measured_source_batch_sizes)) != 1:
        raise ValueError(
            "Only the final dumped source batch may be non-full; measured source "
            f"batch sizes were {measured_source_batch_sizes}"
        )
    if source_batch_sizes[-1] > measured_source_batch_sizes[0]:
        raise ValueError(
            "The final dumped source batch cannot be larger than preceding full "
            f"batches: {source_batch_sizes[-1]} versus "
            f"{measured_source_batch_sizes[0]}"
        )

    warmup_case = input_cases[-1]
    measured_input_cases = _rebatch_input_cases(input_cases[:-1], batch_size)
    if not measured_input_cases:
        raise RuntimeError("No full measured batches were generated")
    return warmup_case, measured_input_cases


def _validate_cache_phase_user_ids(
    warmup_case: InputCase, measured_input_cases: Sequence[InputCase]
) -> tuple[int, int]:
    warmup_user_ids = warmup_case[1][3]
    measured_user_ids = np.concatenate(
        [input_case[3] for _, input_case in measured_input_cases]
    )
    unique_measured_ids, measured_id_counts = np.unique(
        measured_user_ids, return_counts=True
    )
    duplicate_ids = unique_measured_ids[measured_id_counts > 1]
    if duplicate_ids.size:
        raise ValueError(
            "Measured requests contain repeated user IDs, so the first pass "
            "cannot be an all-KV-cache-miss pass. First repeated IDs: "
            f"{duplicate_ids[:8].tolist()}"
        )

    overlapping_ids = np.intersect1d(warmup_user_ids, unique_measured_ids)
    if overlapping_ids.size:
        raise ValueError(
            "Warmup and measured requests share user IDs, so warmup would "
            "populate cache entries used by the first measured pass. First "
            f"overlapping IDs: {overlapping_ids[:8].tolist()}"
        )
    return int(warmup_user_ids.size), int(measured_user_ids.size)


def _offset_input_case_user_ids(
    input_cases: Sequence[InputCase], user_id_offset: int
) -> list[InputCase]:
    if user_id_offset == 0:
        return list(input_cases)

    adjusted_input_cases = []
    for batch_index, input_case in input_cases:
        user_ids = input_case[3]
        if not np.issubdtype(user_ids.dtype, np.signedinteger):
            raise ValueError(
                f"user_ids must use a signed integer dtype: {user_ids.dtype}"
            )
        dtype_limits = np.iinfo(user_ids.dtype)
        adjusted_min = int(user_ids.min()) + user_id_offset
        adjusted_max = int(user_ids.max()) + user_id_offset
        if adjusted_min < dtype_limits.min or adjusted_max > dtype_limits.max:
            raise OverflowError(
                f"user_id offset {user_id_offset} exceeds {user_ids.dtype} range"
            )
        adjusted_input_case = list(input_case)
        adjusted_input_case[3] = user_ids + user_id_offset
        adjusted_input_cases.append((batch_index, adjusted_input_case))
    return adjusted_input_cases


def _build_cache_measurement_sets(
    warmup_case: InputCase, measured_input_cases: Sequence[InputCase]
) -> list[tuple[int, int, list[InputCase]]]:
    all_original_user_ids = np.concatenate(
        [
            warmup_case[1][3],
            *[input_case[3] for _, input_case in measured_input_cases],
        ]
    )
    user_id_stride = (
        int(all_original_user_ids.max()) - int(all_original_user_ids.min()) + 1
    )
    return [
        (
            set_index,
            (set_index - 1) * user_id_stride,
            _offset_input_case_user_ids(
                measured_input_cases, (set_index - 1) * user_id_stride
            ),
        )
        for set_index in range(1, CACHE_MEASUREMENT_SET_COUNT + 1)
    ]


def _run_input_cases(
    client,
    httpclient,
    model_name: str,
    input_cases: list[InputCase],
    outputs,
    *,
    phase: str,
    run_index: int,
    cache_set_index: int,
    user_id_offset: int,
    request_sequence: int,
    profile_records: list[dict[str, Any]],
):
    result = None
    total_latency_ns = 0
    for batch_index, input_case in input_cases:
        batch_size = _logical_batch_size(input_case)
        request_id = (
            f"hstu-aoti-{phase}-s{cache_set_index}-r{run_index}-"
            f"bs{batch_size}-b{batch_index:06d}-q{request_sequence:06d}"
        )
        input_bytes = sum(array.nbytes for array in input_case)
        start_ns = time.perf_counter_ns()
        try:
            result = client.infer(
                model_name,
                inputs=_make_inputs(httpclient, input_case),
                outputs=outputs,
                request_id=request_id,
            )
        except Exception:
            print(f"Request failed for batch_{batch_index:06d}")
            raise
        latency_ns = time.perf_counter_ns() - start_ns
        total_latency_ns += latency_ns
        profile_records.append(
            {
                "request_sequence": request_sequence,
                "request_id": request_id,
                "phase": phase,
                "run_index": run_index,
                "cache_set_index": cache_set_index,
                "user_id_offset": user_id_offset,
                "batch_index": batch_index,
                "batch_size": batch_size,
                "client_latency_ns": latency_ns,
                "client_latency_per_request_ns": latency_ns / batch_size,
                "input_bytes": input_bytes,
            }
        )
        request_sequence += 1
    return result, request_sequence, total_latency_ns


def _write_profile_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as profile_file:
        for record in records:
            profile_file.write(json.dumps(record, sort_keys=True))
            profile_file.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay dumped HSTU KV-cache batches to Triton."
    )
    parser.add_argument(
        "--dump_dir",
        type=Path,
        default=SCRIPT_DIR / "export_test_dump",
        help="Directory containing batch_000000_*.pt dump files.",
    )
    parser.add_argument("--url", type=str, default="localhost:8000")
    parser.add_argument("--model_name", type=str, default="hstu_gr_ranking_kvcache")
    parser.add_argument(
        "--batch_size",
        type=int,
        choices=SUPPORTED_BATCH_SIZES,
        default=2,
        help="Logical HSTU batch size for measured Triton requests.",
    )
    parser.add_argument("--post_warmup_sleep_seconds", type=float, default=1.0)
    parser.add_argument(
        "--print_triton_request_count_only",
        action="store_true",
        help=(
            "Print the number of Triton calls in the two warmups and two "
            "measured cache phases, then exit without connecting to Triton."
        ),
    )
    parser.add_argument(
        "--profile_jsonl",
        type=Path,
        default=None,
        help=(
            "Optional JSONL output with one client latency and Triton request ID "
            "per Triton call. Use this with Triton's TIMESTAMPS trace."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_cases = _load_input_cases(args.dump_dir)
    warmup_case, measured_input_cases = _prepare_request_plan(
        input_cases, args.batch_size
    )
    warmup_user_count, measured_user_count = _validate_cache_phase_user_ids(
        warmup_case, measured_input_cases
    )
    cache_measurement_sets = _build_cache_measurement_sets(
        warmup_case, measured_input_cases
    )
    triton_request_count = WARMUP_COUNT + (
        2 * CACHE_MEASUREMENT_SET_COUNT * len(measured_input_cases)
    )
    if args.print_triton_request_count_only:
        print(triton_request_count)
        return 0

    import tritonclient.http as httpclient

    outputs = [httpclient.InferRequestedOutput("OUTPUT__0")]

    client = httpclient.InferenceServerClient(url=args.url)
    profile_records: list[dict[str, Any]] = []
    request_sequence = 0

    warmup_batch_size = _logical_batch_size(warmup_case[1])
    print(
        f"Loaded {len(input_cases)} source batches from {args.dump_dir}; "
        f"generated {len(measured_input_cases)} full batch_size={args.batch_size} "
        "measured batches"
    )
    print(
        f"Validated cache phases: {measured_user_count} unique measured user IDs "
        f"are disjoint from {warmup_user_count} warmup user IDs"
    )
    for warmup_index in range(1, WARMUP_COUNT + 1):
        _, request_sequence, _ = _run_input_cases(
            client,
            httpclient,
            args.model_name,
            [warmup_case],
            outputs,
            phase="warmup",
            run_index=warmup_index,
            cache_set_index=0,
            user_id_offset=0,
            request_sequence=request_sequence,
            profile_records=profile_records,
        )
        print(
            f"Warmup {warmup_index}/{WARMUP_COUNT}: sent final source batch "
            f"with logical batch_size={warmup_batch_size}; excluded from "
            "measured runs"
        )
    if args.post_warmup_sleep_seconds > 0:
        time.sleep(args.post_warmup_sleep_seconds)
        print(f"Slept {args.post_warmup_sleep_seconds:.3f} seconds after warmup")

    result = None
    for cache_set_index, user_id_offset, cache_set_input_cases in (
        cache_measurement_sets
    ):
        print(
            f"Cache measurement set {cache_set_index}/"
            f"{CACHE_MEASUREMENT_SET_COUNT}: user_id_offset={user_id_offset}"
        )
        for run_index, (phase, label) in enumerate(
            (
                ("no_kvcache", "KV cache miss"),
                ("with_kvcache", "GPU KV cache hit"),
            ),
            start=1,
        ):
            phase_start_ns = time.perf_counter_ns()
            result, request_sequence, _ = _run_input_cases(
                client,
                httpclient,
                args.model_name,
                cache_set_input_cases,
                outputs,
                phase=phase,
                run_index=run_index,
                cache_set_index=cache_set_index,
                user_id_offset=user_id_offset,
                request_sequence=request_sequence,
                profile_records=profile_records,
            )
            phase_e2e_ns = time.perf_counter_ns() - phase_start_ns
            batch_count = len(cache_set_input_cases)
            logical_request_count = batch_count * args.batch_size
            e2e_seconds = phase_e2e_ns / 1_000_000_000
            latency_ms_per_triton_call = phase_e2e_ns / batch_count / 1_000_000
            latency_ms_per_logical_request = (
                phase_e2e_ns / logical_request_count / 1_000_000
            )
            print(
                f"Set {cache_set_index} Run {run_index} "
                f"({label}, batch_size={args.batch_size}): "
                f"{e2e_seconds:.6f} seconds E2E; "
                f"{latency_ms_per_triton_call:.3f} ms/Triton call "
                f"over {batch_count} calls; "
                f"{latency_ms_per_logical_request:.3f} ms/logical request "
                f"over {batch_count} batches x {args.batch_size} "
                f"= {logical_request_count} logical requests"
            )

    if result is None:
        raise RuntimeError("No Triton requests were sent")

    if args.profile_jsonl is not None:
        _write_profile_records(args.profile_jsonl, profile_records)
        print(
            f"Wrote {len(profile_records)} per-Triton-call client timings to "
            f"{args.profile_jsonl}"
        )

    # logits = result.as_numpy("OUTPUT__0")
    # print(f"OUTPUT__0 logits: shape={logits.shape}, dtype={logits.dtype}")
    # print(logits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
