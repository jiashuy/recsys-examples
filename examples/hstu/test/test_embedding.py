# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import inspect
import sys

import pytest
import torch
from configs.inference_config import EmbeddingBackend, InferenceEmbeddingConfig
from modules.inference_embedding import InferenceEmbedding
from modules.nve_compat import (
    imported_nve_generation,
    needs_legacy_embedding_lookup_fake_override,
)
from modules.nve_embeddingcollection import InferenceNVEEmbeddingCollection
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("max_seq_len", [20, 50, 100])
@pytest.mark.parametrize("embedding_dim", [512])
@pytest.mark.parametrize("action_vocab_size", [256])
@pytest.mark.parametrize("item_vocab_size", [10000])
@pytest.mark.parametrize(
    "embedding_backend",
    [
        EmbeddingBackend.NVEMB,
    ],
)
def test_embedding(
    batch_size,
    max_seq_len,
    embedding_dim,
    action_vocab_size,
    item_vocab_size,
    embedding_backend,
):
    if (
        embedding_backend == EmbeddingBackend.NVEMB
        and InferenceNVEEmbeddingCollection is None
    ):
        pytest.skip("NV-Embeddings is not installed.")

    embeddding_configs = [
        InferenceEmbeddingConfig(
            feature_names=["act_feat"],
            table_name="act",
            vocab_size=action_vocab_size,
            dim=embedding_dim,
            use_dynamicemb=False,
        ),
        InferenceEmbeddingConfig(
            feature_names=["item_feat"],
            table_name="item",
            vocab_size=item_vocab_size,
            dim=embedding_dim,
            use_dynamicemb=True,
        ),
    ]

    embedding_collection = InferenceEmbedding(embeddding_configs, embedding_backend)
    embedding_collection = embedding_collection.to(torch.device("cuda:0"))

    act_features_lengths = torch.randint(
        max_seq_len, (batch_size,), device=torch.device("cuda:0")
    )
    item_features_lengths = torch.randint(
        max_seq_len, (batch_size,), device=torch.device("cuda:0")
    )
    act_features = torch.randint(
        action_vocab_size - 1, (torch.sum(act_features_lengths),)
    )
    item_features = torch.randint(
        item_vocab_size - 1, (torch.sum(item_features_lengths),)
    )

    features = KeyedJaggedTensor.from_lengths_sync(
        keys=["act_feat", "item_feat"],
        values=torch.concat([act_features, item_features]).to(torch.device("cuda:0")),
        lengths=torch.concat([act_features_lengths, item_features_lengths])
        .to(torch.device("cuda:0"))
        .long(),
    ).to(torch.device("cuda:0"))

    embeddings = embedding_collection(features)

    features_dict = features.to_dict()
    for key, item in embeddings.items():
        assert torch.allclose(item.lengths(), features_dict[key].lengths())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_nve_hierarchical_nvhashmap_lookup() -> None:
    if InferenceNVEEmbeddingCollection is None:
        pytest.skip("NV-Embeddings is not installed.")

    collection = InferenceNVEEmbeddingCollection(
        configs=[
            EmbeddingConfig(
                name="table",
                num_embeddings=1024,
                embedding_dim=4,
                feature_names=["feature"],
                data_type=torch.float32,
            )
        ],
        device=torch.device("cuda", 0),
        use_gpu_only=False,
        gpu_cache_ratio=1.0,
        sparse_shareables=None,
    )
    embedding = collection.embeddings["table"]
    parameter_server = getattr(
        embedding, "remote_interface", getattr(embedding, "storage", None)
    )
    assert parameter_server is not None

    keys_cpu = torch.tensor([3, 7], dtype=torch.int64)
    values_cpu = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    parameter_server.insert(keys_cpu, values_cpu)
    features = KeyedJaggedTensor.from_lengths_sync(
        keys=["feature"],
        values=keys_cpu.cuda(),
        lengths=torch.tensor([2], dtype=torch.int64, device="cuda"),
    )
    output = collection(features)["feature"].values()
    torch.testing.assert_close(output.cpu(), values_cpu)


def test_embedding_lookup_fake_is_generation_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if InferenceNVEEmbeddingCollection is None:
        pytest.skip("NV-Embeddings is not installed.")

    generation = imported_nve_generation()
    calls = []
    original_register_fake = torch.library.register_fake

    def recording_register_fake(op, *args, **kwargs):
        if op == "nve_ops::embedding_lookup":
            calls.append(kwargs.copy())
        return original_register_fake(op, *args, **kwargs)

    monkeypatch.setattr(torch.library, "register_fake", recording_register_fake)
    sys.modules.pop("modules.exportable_embedding", None)
    exportable_embedding = importlib.import_module("modules.exportable_embedding")

    if generation == "26.05":
        assert needs_legacy_embedding_lookup_fake_override()
        assert calls == [{"allow_override": True}]
        dynamic_size_calls = []

        class FakeContext:
            def new_dynamic_size(self):
                dynamic_size_calls.append(True)
                return 7

        monkeypatch.setattr(torch.library, "get_ctx", lambda: FakeContext())
        fake_output = exportable_embedding._recsys_nve_2605_embedding_lookup_fake(
            torch.zeros(3, dtype=torch.int64), 0
        )
        assert fake_output.shape == (3, 7)
        assert fake_output.dtype == torch.float32
        assert dynamic_size_calls == [True]
    else:
        assert not needs_legacy_embedding_lookup_fake_override()
        assert calls == []
        assert not hasattr(
            exportable_embedding, "_recsys_nve_2605_embedding_lookup_fake"
        )
        import pynve.torch as pynve_torch

        assert list(
            inspect.signature(pynve_torch._embedding_lookup_fake).parameters
        ) == ["marker", "keys", "embedding_size", "dtype"]
