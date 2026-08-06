# HSTU Inference Benchmarks

This benchmark note summarizes the checked-in HSTU ranking inference paths that
use the Triton Python backend, PyTorch AOTInductor (AOTI), native C++ replay,
Triton Server, and the KV-cache runtime.

Covered paths:

- [Triton Python-backend client](../../inference/triton/hstu_model/client.py)
- [PyTorch export, no cache](../export_inference_gr_ranking.py)
- [PyTorch export, with KV cache](../export_inference_gr_ranking_kvcache.py)
- C++ Torch replay, no cache:
`../cpp_inference/build/inference_hstu_gr_ranking_exported_model`
- C++ Torch replay, with KV cache:
`../cpp_inference/build/inference_hstu_gr_ranking_kvcache_exported_model`
- [Triton AOTI deployment
config](../triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt)
- [Triton AOTI request replay
client](../test_tritonserver_aoti_hstu_model.py)



## Benchmark results

The tables below report single-GPU KuaiRand-1K ranking measurements. Unless
otherwise specified, all results use the same model and dataset configuration.

### 1. No-cache HSTU model with Torch C++ runtime

These results are emitted by the no-cache
[export script](../export_inference_gr_ranking.py), which exports the model,
loads the packaged AOTI artifact, and compares Python runtime time against the
native C++ replay executable on the KuaiRand-1K evaluation data.

- 3-Layer HSTU Model (See [below](./README.md#model-structure) for details)


| Hardware                                   | Python runtime per-request latency (ms) | C++ runtime per-request latency (ms) | C++ speedup |
| ------------------------------------------ | --------------------------------------- | ------------------------------------ | ----------- |
| L20                                        | 6.031                                   | 3.708                                | **1.63x**   |
| L40                                        | 4.866                                   | 2.921                                | **1.66x**   |
| L40S                                       | 4.605                                   | 2.419                                | **1.90x**   |
| RTX PRO 6000 Blackwell Workstation Edition | 2.809                                   | 1.950                                | **1.44x**   |




### 2. Triton Server backend comparison

The PyTorch AOTI results use the Triton
[AOTI model config](../triton_aoti/hstu_gr_ranking_kvcache/config.pbtxt).

- Hardware: **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**



#### Runtime and backend comparison

Benchmark with Triton batch size = 2.

- 3-Layer HSTU Model

| Runtime or serving path            | Cache state      | Latency per logical request (ms) | Latency Speedup   |
| ---------------------------------- | ---------------- | -------------------------------- | ----------------- |
| Triton Server Python backend       | No cache         | 3.161                            | Baseline          |
| Triton Server PyTorch AOTI backend | No cache         | 2.471                            | **1.28x**         |
| Triton Server PyTorch AOTI backend | GPU KV-cache hit | 1.436                            | **2.20x**         |


- 8-Layer HSTU Model


| Runtime or serving path            | Cache state      | Latency per logical request (ms) | Latency Speedup   |
| ---------------------------------- | ---------------- | -------------------------------- | ----------------- |
| Triton Server Python backend       | No cache         | 5.879                            | Baseline          |
| Triton Server PyTorch AOTI backend | No cache         | 5.156                            | **1.14x**         |
| Triton Server PyTorch AOTI backend | GPU KV-cache hit | 2.467                            | **2.38x**         |




#### PyTorch AOTI backend by batch size

- 3-Layer HSTU Model


| Batch Size | No-cache latency per logical request (ms) | GPU KV-cache-hit latency per logical request (ms) | GPU KV-cache-hit speedup |
| ---------- | ----------------------------------------- | ------------------------------------------------- | ------------------------ |
| 2          | 2.471                                     | 1.436                                             | **1.72x**                |
| 4          | 1.956                                     | 0.763                                             | **2.56x**                |
| 8          | 1.890                                     | 0.423                                             | **4.47x**                |


- 8-Layer HSTU Model


| Batch Size | No-cache latency per logical request (ms) | GPU KV-cache-hit latency per logical request (ms) | GPU KV-cache-hit speedup |
| ---------- | ----------------------------------------- | ------------------------------------------------- | ------------------------ |
| 2          | 5.156                                     | 2.467                                             | **2.09x**                |
| 4          | 4.406                                     | 1.267                                             | **3.48x**                |
| 8          | 4.020                                     | 0.678                                             | **5.93x**                |




### Notes:



#### Model structure

The model settings come from the KuaiRand-1K ranking
[gin configuration](../../inference/configs/kuairand_1k_inference_ranking.gin).
The effective sequence capacity follows the
[AOTI exporter](../export_inference_gr_ranking_kvcache.py).


| Model property                             | Value                               |
| ------------------------------------------ | ----------------------------------- |
| HSTU layers                                | 3 / 8                               |
| Hidden size                                | 512                                 |
| Attention heads                            | 4                                   |
| Head dimension (`kv_channels`)             | 128                                 |
| Model and KV-cache dtype                   | BF16                                |
| Maximum history sequence length            | 4096 per item/action history stream |
| Maximum candidate sequence length          | 100                                 |
| Contextual features                        | 6                                   |
| Effective sequence length before alignment | 8298 (`2 * 4096 + 100 + 6`)         |
| Exported maximum sequence length           | 8320 (aligned to 32 tokens)         |


The gin file does not override `NetworkArgs.dtype_str`; its BF16 default is
propagated by the exporter and checked against the KV-cache runtime
configuration.

#### Triton benchmark protocol

- The client uses KuaiRand-1K evaluation data.
- A fresh Triton process and a fresh FlexKV process are started for each batch
size.
- Triton's scheduler-level batching is disabled due to jagged data input format.
One Triton call contains one logical batch, so latency per logical request is
the pass E2E time divided by `number of Triton calls * logical batch size`.
- Dataset loading, validation, rebatching, user-ID generation, server
startup, warmup, and the post-warmup sleep are excluded from measured time.

