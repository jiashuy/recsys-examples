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

import warnings
from collections import deque
from itertools import accumulate
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple

import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.training.batched_dynamicemb_function import (
    DynamicEmbeddingFunction,
    PrefetchState,
    dynamicemb_eval_forward,
    dynamicemb_prefetch,
)
from dynamicemb.storage.chained_storage import ChainedHybridStorage, ChainedStorage
from dynamicemb.checkpoint.checkpointer import Checkpointer, ShardInfo

# The inference module imports the file-name helpers from here.
from dynamicemb.checkpoint.paths import (  # noqa: F401
    encode_checkpoint_file_path,
    encode_counter_checkpoint_file_path,
    encode_meta_json_file_path,
    find_files,
    get_loading_files,
)
from dynamicemb.dynamicemb_config import (
    DynamicEmbEvictStrategy,
    DynamicEmbPoolingMode,
    DynamicEmbScoreStrategy,
    DynamicEmbTableOptions,
    ReplayContent,
    get_eviction_score_strategy,
    score_strategy_has_timestamp_column,
    warning_for_cstm_score,
)
from dynamicemb.initializer import MultiTableInitializer
from dynamicemb.storage.cache import flush_cache
from dynamicemb.storage.row_exchange import RowExchange
from dynamicemb.storage.memory_report import print_memory_consume
from dynamicemb.storage.storage_layout import build_storage
from dynamicemb.training.lookup_layout import EmbeddingLayout
from dynamicemb.optimizer import (
    AdaGradDynamicEmbeddingOptimizer,
    AdamDynamicEmbeddingOptimizer,
    BaseDynamicEmbeddingOptimizer,
    DynamicEmbOptimType,
    EmbOptimType,
    FTRLDynamicEmbeddingOptimizer,
    OptimizerArgs,
    OptimType,
    RowWiseAdaGradDynamicEmbeddingOptimizer,
    SGDDynamicEmbeddingOptimizer,
)
from dynamicemb.types import Cache, ReplayStats, Storage
from dynamicemb_extensions import device_timestamp
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    BoundsCheckMode,
    CounterBasedRegularizationDefinition,
    CowClipDefinition,
    WeightDecayMode,
)
from torch import Tensor, nn  # usort:skip

if TYPE_CHECKING:
    from dynamicemb.incremental_dump import DeltaDumpResult


class BatchedDynamicEmbeddingTablesV2(nn.Module):
    """
    Dynamic Embedding uses a GPU-optimized scored hash table backend.
    Looks up one or more dynamic embedding tables. The module is application for training.

    Its optional to fuse the optimizer with the backward operator by parameter *update_grads_explicitly*.
    """

    optimizer_args: OptimizerArgs

    def __init__(
        self,
        table_options: List[DynamicEmbTableOptions],
        table_names: Optional[List[str]] = None,
        feature_table_map: Optional[List[int]] = None,  # [T]
        use_index_dedup: bool = False,
        prefetch_pipeline: bool = False,  #  we set the arg name same as FBGEMM TBE to align with it
        pooling_mode: DynamicEmbPoolingMode = DynamicEmbPoolingMode.SUM,
        output_dtype: torch.dtype = torch.float32,
        device: torch.device = None,
        enforce_hbm: bool = False,  # place all weights/momentums in HBM when using cache
        bounds_check_mode: BoundsCheckMode = BoundsCheckMode.WARNING,
        optimizer: OptimType = EmbOptimType.SGD,
        # General Optimizer args
        stochastic_rounding: bool = True,
        gradient_clipping: bool = False,
        max_gradient: float = 1.0,
        max_norm: float = 0.0,
        learning_rate: float = 0.01,
        # used by EXACT_ADAGRAD, EXACT_ROWWISE_ADAGRAD, EXACT_ROWWISE_WEIGHTED_ADAGRAD, LAMB, and ADAM only
        # NOTE that default is different from nn.optim.Adagrad default of 1e-10
        eps: float = 1.0e-8,
        # used by EXACT_ADAGRAD, EXACT_ROWWISE_ADAGRAD, EXACT_ROWWISE_WEIGHTED_ADAGRAD
        # and FTRL only. For FTRL this seeds the squared-gradient accumulator;
        # prefer ftrl_beta to bound the first steps, see its note below.
        initial_accumulator_value: float = 0.0,
        # used by FTRL only; ftrl_beta / l1_reg / l2_reg are the beta, lambda1
        # and lambda2 of McMahan et al. 2013
        learning_rate_power: float = -0.5,
        ftrl_beta: float = 0.0,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        momentum: float = 0.9,  # used by LARS-SGD
        # EXACT_ADAGRAD, SGD, EXACT_SGD do not support weight decay
        # LAMB, ADAM, PARTIAL_ROWWISE_ADAM, PARTIAL_ROWWISE_LAMB, LARS_SGD support decoupled weight decay
        # EXACT_ROWWISE_WEIGHTED_ADAGRAD supports L2 weight decay
        # EXACT_ROWWISE_ADAGRAD support both L2 and decoupled weight decay (via weight_decay_mode)
        weight_decay: float = 0.0,
        weight_decay_mode: WeightDecayMode = WeightDecayMode.NONE,
        eta: float = 0.001,  # used by LARS-SGD,
        beta1: float = 0.9,  # used by LAMB and ADAM
        beta2: float = 0.999,  # used by LAMB and ADAM
        counter_based_regularization: Optional[
            CounterBasedRegularizationDefinition
        ] = None,  # used by Rowwise Adagrad
        cowclip_regularization: Optional[
            CowClipDefinition
        ] = None,  # used by Rowwise Adagrad
        shard_ranks: Optional[List[int]] = None,
        # FTRL only, G-FTRL (Ni et al., CIKM 2019): lambda, a row is a group.
        # Kept last so it cannot shift any positional argument;
        # group_lasso_reg = 0 is plain FTRL.
        group_lasso_reg: float = 0.0,
        # TO align with FBGEMM TBE
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        assert len(table_options) >= 1
        table_option = table_options[0]
        for other_option in table_options:
            assert (
                table_option == other_option
            ), "All tables must match in grouped keys."
        self._dynamicemb_options = table_options
        # The ranks that hold these tables, in shard order: the world for a
        # row-wise table, one node for a table-row-wise one. Everything that
        # names a shard -- checkpoint files, the load-time key filter, the delta
        # meta -- derives from this rather than from the global process group.
        self._shard = ShardInfo(shard_ranks, table_names)
        self.initializer_args = table_option.initializer_args
        self.index_type = table_option.index_type
        self.embedding_dtype = table_option.embedding_dtype
        self.output_dtype = output_dtype
        self.pooling_mode = pooling_mode
        self.use_index_dedup = use_index_dedup
        self._enable_prefetch = prefetch_pipeline
        self.prefetch_stream = None
        # Keys of the steps in flight, per table; only a cache looks at it.
        self._prefetch_outstanding_keys = torch.zeros(
            len(self._dynamicemb_options), dtype=torch.int64
        )
        self._table_names = table_names
        self.bounds_check_mode_int: int = bounds_check_mode.value
        self._create_score()
        # Every table of a module has the same eviction strategy: it is part of
        # what they were grouped on, and one physical table has one score
        # layout. The admission strategy is materialized further down, once the
        # device it allocates on is known.
        self._evict_strategy = self._dynamicemb_options[0].evict_strategy.value
        if device is not None:
            self.device_id = int(str(device)[-1])
        else:
            assert torch.cuda.is_available(), "No available CUDA device."
            self.device_id = torch.cuda.current_device()

        if table_option.device_id is None:
            for option in self._dynamicemb_options:
                option.device_id = self.device_id
        self.dims: List[int] = [option.dim for option in self._dynamicemb_options]
        # Sequence mode requires uniform embedding dim because the output is
        # [N, D].  Pooling mode supports mixed dims natively via D_offsets.
        if pooling_mode == DynamicEmbPoolingMode.NONE:
            assert all(d == self.dims[0] for d in self.dims), (
                f"Sequence mode requires uniform embedding dim, got {set(self.dims)}. "
                "Tables with different dims are automatically split into separate "
                "BatchedDynamicEmbeddingTablesV2 instances by the planner."
            )

        # physical table number.
        T_ = len(self._dynamicemb_options)
        assert T_ > 0
        self.feature_table_map: List[int] = (
            feature_table_map if feature_table_map is not None else list(range(T_))
        )
        # logical table number.
        T = len(self.feature_table_map)
        assert T_ <= T
        table_has_feature = [False] * T_
        for t in self.feature_table_map:
            table_has_feature[t] = True
        assert all(table_has_feature), "Each table must have at least one feature!"

        feature_dims = [self.dims[t] for t in self.feature_table_map]
        D_offsets = [0] + list(accumulate(feature_dims))
        self.total_D: int = D_offsets[-1]
        self.max_D: int = max(self.dims)
        # The shared dim, or None when the tables differ. Mirrors the condition
        # the D_offsets buffer below is registered on.
        self.common_D: Optional[int] = (
            self.dims[0] if self.max_D == min(self.dims) else None
        )
        # Whether the pooled kernels may use their vectorized path; see
        # EmbeddingLayout.feature_dims_vec4.  Every table has at least one
        # feature, so checking the table dims covers the feature dims.
        self.feature_dims_vec4: bool = all(d % 4 == 0 for d in self.dims)

        # Per-feature cumulative dimension offsets, registered on GPU for use
        # by multi-dim pooling kernels.  Only needed when tables have mixed
        # embedding dimensions; uniform-dim pooling uses a simpler path.
        if self.max_D > min(self.dims):
            self.register_buffer(
                "D_offsets_t",
                torch.tensor(
                    D_offsets,
                    device=torch.device(self.device_id),
                    dtype=torch.int32,
                ),
            )
        else:
            self.register_buffer("D_offsets_t", None)

        self.feature_num = len(self.feature_table_map)
        # TODO:deal with shuffeld feature_table_map
        self.table_offsets_in_feature: List[int] = []
        old_table_id = -1
        for idx, table_id in enumerate(self.feature_table_map):
            if table_id != old_table_id:
                self.table_offsets_in_feature.append(idx)
                old_table_id = table_id
        self.table_offsets_in_feature.append(self.feature_num)
        self.feature_offsets = torch.tensor(
            self.table_offsets_in_feature,
            device=torch.device(self.device_id),
            dtype=torch.int64,
        )

        for option in self._dynamicemb_options:
            if option.init_capacity is None:
                option.init_capacity = option.max_capacity

        self._optimizer: BaseDynamicEmbeddingOptimizer = self._create_optimizer(
            optimizer,
            stochastic_rounding,
            gradient_clipping,
            max_gradient,
            max_norm,
            learning_rate,
            eps,
            initial_accumulator_value,
            beta1,
            beta2,
            weight_decay,
            eta,
            momentum,
            weight_decay_mode,
            counter_based_regularization,
            cowclip_regularization,
            learning_rate_power,
            ftrl_beta,
            l1_reg,
            l2_reg,
            group_lasso_reg,
            table_option.optimizer_state_dtype,
        )
        self._storage_externel = table_option.external_storage is not None
        self._create_cache_storage()
        self._create_initializers()
        self._create_admitter()
        self._prefetch_states: Deque[PrefetchState] = deque()
        self._checkpointer = Checkpointer(
            self._storage,
            self._cache,
            self.flush,
            self._table_names,
            self._dynamicemb_options,
            self._shard,
            self._scores,
            self._admission_counter,
        )

        # TODO:1->10
        self._empty_tensor = nn.Parameter(
            torch.empty(
                10,
                requires_grad=True,
                device=torch.device(self.device_id),
                dtype=self.embedding_dtype,
            )
        )

    def _create_cache_storage(self) -> None:
        self._cache: Optional[Cache] = None
        self._exchange: Optional[RowExchange] = None
        self._caching = any(option.caching for option in self._dynamicemb_options)

        for option in self._dynamicemb_options:
            if option.training:
                assert (
                    self._optimizer_type != EmbOptimType.NONE
                ), "Optimizer type must be set for training mode."
            else:
                if self._optimizer_type != EmbOptimType.NONE:
                    self._optimizer_type = EmbOptimType.NONE
                    warnings.warn(
                        "Set optimizer type to NONE as not on training mode.",
                        UserWarning,
                    )

        bundle = build_storage(
            self._dynamicemb_options, self._optimizer, self.device_id
        )
        self._storage = bundle.storage
        self._cache = bundle.cache
        self._exchange = bundle.exchange
        self._storage_layout = bundle.layout

        print_memory_consume(
            self._table_names,
            self._storage,
            self._cache,
            self._storage_layout,
            self._dynamicemb_options,
            self._optimizer,
            self.device_id,
            self._optimizer_type,
        )

    def _create_initializers(self) -> None:
        """One initializer per module, not per table.

        The tables share a value buffer, so a row is addressed by its position
        in that buffer; a MultiTableInitializer turns that position into the
        owning table's parameters. Tables are only grouped by initializer mode,
        so their parameters really can differ -- and do by default, since an
        unbounded UNIFORM resolves against each table's num_embeddings.
        """
        device = torch.device(f"cuda:{self.device_id}")
        self._initializer = MultiTableInitializer.create(
            [option.initializer_args for option in self._dynamicemb_options], device
        )
        self._eval_initializer = MultiTableInitializer.create(
            [option.eval_initializer_args for option in self._dynamicemb_options],
            device,
        )

    def _create_admitter(self) -> None:
        """Turn the tables' admission configurations into the one this runs.

        The configurations the caller wrote are inert; this is where the
        admitter's device state -- a counter's hash table, an initializer's
        per-table parameters -- is allocated. Whatever of it has to be
        checkpointed and accounted for comes back through ``state()``.
        """
        table_strategies = [
            option.admit_strategy for option in self._dynamicemb_options
        ]
        if all(strategy is None for strategy in table_strategies):
            self._admitter = None
            self._admission_counter = None
            return
        if any(strategy is None for strategy in table_strategies):
            # Unreachable while admit_strategy is part of the grouping key;
            # say so rather than let a None reach the forward.
            raise ValueError(
                "Either every table of a module has an admission strategy or "
                "none does."
            )
        self._admitter = type(table_strategies[0]).create_admitter(
            table_strategies, torch.device(f"cuda:{self.device_id}")
        )
        self._admission_counter = self._admitter.state()
        if isinstance(self._storage, ChainedStorage):
            self._storage.set_admitter(self._admitter)

    def _create_optimizer(
        self,
        optimizer_type: OptimType,
        stochastic_rounding: bool,
        gradient_clipping: bool,
        max_gradient: float,
        max_norm: float,
        learning_rate: float,
        eps: float,
        initial_accumulator_value: float,
        beta1: float,
        beta2: float,
        weight_decay: float,
        eta: float,
        momentum: float,
        weight_decay_mode: WeightDecayMode,
        counter_based_regularization: Optional[CounterBasedRegularizationDefinition],
        cowclip_regularization: Optional[CowClipDefinition],
        learning_rate_power: float,
        ftrl_beta: float,
        l1_reg: float,
        l2_reg: float,
        group_lasso_reg: float,
        optimizer_state_dtype: Optional[torch.dtype],
    ) -> BaseDynamicEmbeddingOptimizer:
        self._optimizer_type = optimizer_type
        self.stochastic_rounding = stochastic_rounding

        self.weight_decay_mode = weight_decay_mode
        if (weight_decay_mode == WeightDecayMode.COUNTER) != (
            counter_based_regularization is not None
        ):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COUNTER together with valid counter_based_regularization"
            )
        if (weight_decay_mode == WeightDecayMode.COWCLIP) != (
            cowclip_regularization is not None
        ):
            raise AssertionError(
                "Need to set weight_decay_mode=WeightDecayMode.COWCLIP together with valid cowclip_regularization"
            )

        self._used_rowwise_adagrad_with_counter: bool = (
            optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD
            and (
                weight_decay_mode in (WeightDecayMode.COUNTER, WeightDecayMode.COWCLIP)
            )
        )

        if counter_based_regularization is None:
            counter_based_regularization = CounterBasedRegularizationDefinition()
        if cowclip_regularization is None:
            cowclip_regularization = CowClipDefinition()
        self._max_counter_update_freq: int = -1
        # Extract parameters from CounterBasedRegularizationDefinition or CowClipDefinition
        # which are passed as entries for OptimizerArgs
        if self._used_rowwise_adagrad_with_counter:
            if self.weight_decay_mode == WeightDecayMode.COUNTER:
                self._max_counter_update_freq = (
                    counter_based_regularization.max_counter_update_freq
                )
                opt_arg_weight_decay_mode = (
                    counter_based_regularization.counter_weight_decay_mode
                )
                counter_halflife = counter_based_regularization.counter_halflife
            else:
                opt_arg_weight_decay_mode = (
                    cowclip_regularization.counter_weight_decay_mode
                )
                counter_halflife = cowclip_regularization.counter_halflife
        else:
            opt_arg_weight_decay_mode = weight_decay_mode
            # Default: -1, no decay applied, as a placeholder for OptimizerArgs
            # which should not be effective when CounterBasedRegularizationDefinition
            # and CowClipDefinition are not used
            counter_halflife = -1

        optimizer_args = OptimizerArgs(
            stochastic_rounding=stochastic_rounding,
            gradient_clipping=gradient_clipping,
            max_gradient=max_gradient,
            max_norm=max_norm,
            learning_rate=learning_rate,
            eps=eps,
            initial_accumulator_value=initial_accumulator_value,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            weight_decay_mode=opt_arg_weight_decay_mode.value,
            eta=eta,
            momentum=momentum,
            counter_halflife=counter_halflife,
            adjustment_iter=counter_based_regularization.adjustment_iter,
            adjustment_ub=counter_based_regularization.adjustment_ub,
            learning_rate_mode=counter_based_regularization.learning_rate_mode.value,
            grad_sum_decay=counter_based_regularization.grad_sum_decay.value,
            tail_id_threshold=counter_based_regularization.tail_id_threshold.val,
            is_tail_id_thresh_ratio=int(
                counter_based_regularization.tail_id_threshold.is_ratio
            ),
            total_hash_size=0,
            weight_norm_coefficient=cowclip_regularization.weight_norm_coefficient,
            lower_bound=cowclip_regularization.lower_bound,
            regularization_mode=weight_decay_mode.value,
            learning_rate_power=learning_rate_power,
            ftrl_beta=ftrl_beta,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            group_lasso_reg=group_lasso_reg,
            optimizer_state_dtype=optimizer_state_dtype,
        )
        self._optimizer_args = optimizer_args

        if optimizer_type == EmbOptimType.SGD:
            optimizer = SGDDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        elif optimizer_type == EmbOptimType.EXACT_SGD:
            optimizer = SGDDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        elif optimizer_type == EmbOptimType.ADAM:
            optimizer = AdamDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        elif optimizer_type == EmbOptimType.EXACT_ADAGRAD:
            optimizer = AdaGradDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        elif optimizer_type == EmbOptimType.EXACT_ROWWISE_ADAGRAD:
            optimizer = RowWiseAdaGradDynamicEmbeddingOptimizer(
                optimizer_args,
                self.embedding_dtype,
            )
        elif optimizer_type == DynamicEmbOptimType.FTRL:
            optimizer = FTRLDynamicEmbeddingOptimizer(
                optimizer_args,
            )
        else:
            raise ValueError(
                f"Not supported optimizer type ,optimizer type = {optimizer_type} {type(optimizer_type)} {optimizer_type.value}."
            )
        optimizer.get_state_dtype(self.embedding_dtype)
        return optimizer

    def split_embedding_weights(self) -> List[Tensor]:
        """
        Returns a list of weights, split by table
        """
        splits = []
        for t, _ in enumerate(self._dynamicemb_options):
            splits.append(
                torch.empty(
                    (1, 1), device=torch.device("cuda"), dtype=self.embedding_dtype
                )
            )
        return splits

    def flush(self) -> None:
        if self._caching and self._cache is not None:
            flush_cache(self._cache, self._storage)

    def reset_cache_states(self) -> None:
        if self._caching and self._cache is not None:
            self._cache.reset()
            self._prefetch_outstanding_keys.zero_()

    @property
    def table_names(self) -> List[str]:
        return self._table_names

    @property
    def optimizer(
        self,
    ) -> BaseDynamicEmbeddingOptimizer:
        return self._optimizer

    @property
    def tables(self) -> Storage:
        return self._storage

    @property
    def cache(self) -> Optional[Cache]:
        return self._cache

    def set_record_cache_metrics(self, record: bool) -> None:
        if self._cache is not None:
            self._cache.set_record_cache_metrics(record)

    def set_learning_rate(self, lr: float) -> None:
        self._optimizer.set_learning_rate(lr)

    @property
    def enable_prefetch(
        self,
    ) -> None:
        return self._enable_prefetch

    @enable_prefetch.setter
    def enable_prefetch(self, value: bool):
        self._enable_prefetch = value

    def forward(
        self,
        indices: Tensor,
        offsets: Tensor,
        frequency_counters: Optional[Tensor] = None,
        pooling_weights: Optional[Tensor] = None,
        feature_requires_grad: Optional[Tensor] = None,
        # 2D tensor of batch size for each rank and feature.
        # Shape (number of features, number of ranks)
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        total_unique_indices: Optional[int] = None,
    ) -> List[Tensor]:
        if indices.dtype != self.index_type:
            indices = indices.to(self.index_type)

        for table_name in self._table_names:
            if table_name not in self._scores.keys():
                raise RuntimeError(
                    f"Must set score for table '{table_name}' whose score_strategy is customized."
                )

        feature_batch_size = offsets.numel() - 1
        assert feature_batch_size > 0, "feature_batch_size must be greater than 0"
        assert (
            feature_batch_size % self.feature_num == 0
        ), "feature_batch_size must be divisible by feature_num"
        batch_size = (
            feature_batch_size // self.feature_num if self.feature_num > 0 else 0
        )
        # Built per call because batch_size comes from the input; see
        # dynamicemb/lookup_layout.py for what the fields mean.
        layout = EmbeddingLayout(
            pooling_mode=self.pooling_mode,
            batch_size=batch_size,
            feature_num=self.feature_num,
            num_keys=indices.numel(),
            total_D=self.total_D,
            max_D=self.max_D,
            common_D=self.common_D,
            D_offsets=self.D_offsets_t,
            feature_dims_vec4=self.feature_dims_vec4,
        )
        if pooling_weights is not None:
            if frequency_counters is not None:
                # Both are fed from the single KJT weights channel: the
                # sequence path reads it as LFU frequency counters, the
                # weighted-pooling path as per-sample weights. One KJT cannot
                # mean both, so receiving both signals a misrouted caller.
                raise ValueError(
                    "pooling_weights and frequency_counters are mutually "
                    "exclusive (both are derived from KeyedJaggedTensor "
                    "weights); got both."
                )
            if self.pooling_mode != DynamicEmbPoolingMode.SUM:
                raise ValueError(
                    "pooling_weights requires pooling_mode=SUM (weighted pooling "
                    f"is only supported for SUM, got {self.pooling_mode})."
                )
            if pooling_weights.dtype != torch.float32:
                raise ValueError(
                    f"pooling_weights must be float32, got {pooling_weights.dtype}."
                )
            if pooling_weights.numel() != indices.numel():
                raise ValueError(
                    "pooling_weights.numel() "
                    f"({pooling_weights.numel()}) must equal indices.numel() "
                    f"({indices.numel()})."
                )
            # The backward returns no gradient for the weights, so autograd
            # would treat theirs as zero and a module producing learned weights
            # would train as if it were detached -- silently, since a None
            # gradient is not an error. Reject that outright rather than let it
            # look like it works. Guarded on grad mode so eval under no_grad()
            # accepts weights that merely happen to carry requires_grad.
            if torch.is_grad_enabled() and pooling_weights.requires_grad:
                raise ValueError(
                    "pooling_weights with requires_grad=True is not supported: "
                    "no gradient is computed for them, so learned per-position "
                    "weights cannot train. Pass detached weights, or compute "
                    "the weight gradient outside this module."
                )

        if not self.training:
            scores = [self._scores[name] for name in self._table_names]
            fused_score = self._reduce_table_scores(scores)
            if isinstance(self._storage, (ChainedStorage, ChainedHybridStorage)):
                self._storage.training = False
                self._storage.set_score(fused_score)
            return dynamicemb_eval_forward(
                indices,
                offsets,
                self._cache,
                self._storage,
                self.feature_offsets,
                self.output_dtype,
                self._eval_initializer,
                layout,
                self._evict_strategy,
                frequency_counters,
                pooling_weights,
            )

        if any([not o.training for o in self._dynamicemb_options]):
            raise RuntimeError(
                "BatchedDynamicEmbeddingTables does not support training when some tables are in eval mode."
            )

        if not self._prefetch_states:
            self.prefetch(indices, offsets, frequency_counters=frequency_counters)
        prefetch_state = self._prefetch_states.popleft()

        res = DynamicEmbeddingFunction.apply(
            prefetch_state,
            offsets,
            self._cache,
            self._storage,
            self.output_dtype,
            self._initializer,
            self._optimizer,
            layout,
            self._admitter,
            self._evict_strategy,
            pooling_weights,
            self._empty_tensor,
        )
        if isinstance(self._storage, (ChainedStorage, ChainedHybridStorage)):
            self._storage.training = False

        return res

    def prefetch(
        self,
        indices: Tensor,
        offsets: Tensor,
        forward_stream: Optional[torch.cuda.Stream] = None,
        batch_size_per_feature_per_rank: Optional[List[List[int]]] = None,
        frequency_counters: Optional[Tensor] = None,
    ) -> None:
        if not self.training:
            return
        if self.prefetch_stream is None and forward_stream is not None:
            self.prefetch_stream = torch.cuda.current_stream()
            assert (
                self.prefetch_stream != forward_stream
            ), "prefetch_stream and forward_stream should not be the same stream"

            current_stream = torch.cuda.current_stream()
            indices.record_stream(current_stream)
            offsets.record_stream(current_stream)

        scores = [self._scores[name] for name in self._table_names]
        fused_score = self._reduce_table_scores(scores)
        if isinstance(self._storage, (ChainedStorage, ChainedHybridStorage)):
            self._storage.training = True
            self._storage.set_score(fused_score)

        self._prefetch_states.append(
            dynamicemb_prefetch(
                indices,
                offsets,
                self._cache,
                self._storage,
                self.feature_offsets,
                self._initializer,
                forward_stream,
                self._evict_strategy,
                frequency_counters,
                self._admitter,
                exchange=self._exchange,
                outstanding_keys_ref=self._prefetch_outstanding_keys
                if self._cache is not None
                else None,
            )
        )
        self._update_score()

    def set_score(
        self,
        named_score: Dict[str, int],
    ) -> None:
        table_names: List[str] = named_score.keys()
        table_scores: List[int] = named_score.values()
        for table_name, table_score in zip(table_names, table_scores):
            if not isinstance(table_score, int):
                raise ValueError(
                    f"Table's score is expect to int but got {type(table_score)}"
                )
            if table_score == 0:
                raise ValueError(f"Can't set table's score to 0.")
            index = self._table_names.index(table_name)
            assert (
                self._dynamicemb_options[index].score_strategy
                == DynamicEmbScoreStrategy.CUSTOMIZED
            ), "Can only set score for table whose score_strategy is DynamicEmbScoreStrategy.CUSTOMIZED."

            if table_name in self._scores and self._scores[table_name] > table_score:
                if warning_for_cstm_score():
                    warnings.warn(
                        f"New set score is less than the old one for table '{table_name}': {table_score} < {self._scores[table_name]}",
                        UserWarning,
                    )
            self._scores[table_name] = table_score

    def get_score(self) -> Dict[str, int]:
        """Return current score per table. For any table whose score strategy
        carries a device-timestamp column (single TIMESTAMP, or a compound strategy
        that includes TIMESTAMP, whose incremental dump thresholds on that column),
        returns device_timestamp(); otherwise returns the stored score."""
        result: Dict[str, int] = {}
        ts: Optional[int] = None
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            if score_strategy_has_timestamp_column(option.score_strategy):
                if ts is None:
                    ts = device_timestamp()
                result[table_name] = ts
            else:
                result[table_name] = self._scores[table_name]
        return result

    def fill_tables(
        self,
        load_factor: float = 0.95,
        tolerance: float = 1e-5,
    ) -> None:
        """
        Raise the occupancy of every table toward the given load factor using random keys.

        Default ``load_factor`` is ``0.95``; values above ``0.95`` are clamped. Keys are
        sampled uniformly from ``[0, 2**63 - 2]`` (``torch.randint`` cannot use
        ``high = 2**63``). Only the index is updated; the rows are not written
        (typically still zeros from allocation). A table counts as done once its
        load factor is within ``tolerance`` of the target.

        Only supported for a single tier (not the hybrid layout).
        """
        if not isinstance(self._storage, ChainedStorage):
            raise TypeError(
                "fill_tables requires a single-tier storage; "
                f"got {type(self._storage).__name__}"
            )
        fused_score = max(self.get_score().values())
        self._storage.set_score(fused_score)
        self._storage.fill_tables(load_factor, tolerance)

    def _create_score(self):
        self._scores: Dict[str, int] = {}
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            # Compound strategies (e.g. (TIMESTAMP, LFU)) are driven by their
            # eviction element: the score value fed to the table and the eviction
            # strategy follow that element (the timestamp column is maintained
            # automatically by the LruLfu policy). Single strategies map to
            # themselves.
            eviction_strategy = get_eviction_score_strategy(option.score_strategy)
            if eviction_strategy == DynamicEmbScoreStrategy.TIMESTAMP:
                option.evict_strategy = DynamicEmbEvictStrategy.LRU
                self._scores[table_name] = 0
            elif eviction_strategy == DynamicEmbScoreStrategy.STEP:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
                self._scores[table_name] = 1
            elif eviction_strategy == DynamicEmbScoreStrategy.CUSTOMIZED:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
                self._scores[table_name] = 0
            elif eviction_strategy == DynamicEmbScoreStrategy.LFU:
                option.evict_strategy = DynamicEmbEvictStrategy.LFU
                self._scores[table_name] = 1
            elif eviction_strategy == DynamicEmbScoreStrategy.NO_EVICTION:
                option.evict_strategy = DynamicEmbEvictStrategy.CUSTOMIZED
                self._scores[table_name] = 0

    def _update_score(self):
        """Only STEP mode updates score; TIMESTAMP/LFU are not used by the underlying table or are constant."""
        for table_name, option in zip(self._table_names, self._dynamicemb_options):
            if option.score_strategy != DynamicEmbScoreStrategy.STEP:
                continue
            old_score = self._scores[table_name]
            max_uint64 = (2**64) - 1
            new_score = old_score + 1
            if new_score > max_uint64:
                warnings.warn(
                    f"Table '{table_name}' 's score({new_score}) is out of range, reset to 0.",
                    UserWarning,
                )
                self._scores[table_name] = 0
            else:
                self._scores[table_name] = new_score

    def _reduce_table_scores(self, scores: List[int]) -> int:
        if len(scores) == 0:
            return 1
        if len(set(scores)) > 1:
            warnings.warn(
                "Found different table scores in fused mode; using the max score.",
                UserWarning,
            )
        return max(scores)

    def set_shard_pg(self, pg: dist.ProcessGroup) -> None:
        """Name the group of ranks that hold this kernel's tables (see
        :meth:`ShardInfo.set_group`)."""
        self._shard.set_group(pg)

    def dump(
        self,
        save_dir: str,
        optim: bool = False,
        counter: bool = False,
        table_names: Optional[List[str]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        """Write the tables to ``save_dir`` (see :meth:`Checkpointer.dump`)."""
        self._checkpointer.dump(save_dir, optim, counter, table_names, pg)

    def load(
        self,
        save_dir: str,
        optim: bool = False,
        counter: bool = False,
        table_names: Optional[List[str]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ):
        """Read the tables back from ``save_dir`` (see :meth:`Checkpointer.load`)."""
        return self._checkpointer.load(save_dir, optim, counter, table_names, pg)

    def export_keys_values(
        self, table_name: str, device: torch.device, batch_size: int = 65536
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """This rank's (keys, embeddings) for one table (see
        :meth:`Checkpointer.export_keys_values`)."""
        return self._checkpointer.export_keys_values(table_name, device, batch_size)

    def pop_evicted_keys(
        self, table_names: Optional[List[str]] = None
    ) -> Dict[str, Tensor]:
        """Return and clear this rank's retained evicted keys (see
        :meth:`Checkpointer.pop_evicted_keys`)."""
        return self._checkpointer.pop_evicted_keys(table_names)

    def pop_erased_keys(
        self, table_names: Optional[List[str]] = None
    ) -> Dict[str, Tensor]:
        """Return and clear the keys an explicit erase removed (see
        :meth:`Checkpointer.pop_erased_keys`)."""
        return self._checkpointer.pop_erased_keys(table_names)

    def incremental_dump(
        self,
        named_thresholds: Dict[str, int],
        pg: Optional[dist.ProcessGroup] = None,
    ) -> "DeltaDumpResult":
        """Dump what crossed the score threshold (see
        :meth:`Checkpointer.incremental_dump`)."""
        return self._checkpointer.incremental_dump(named_thresholds, pg)

    def replay_increment(
        self,
        delta: "DeltaDumpResult",
        content: ReplayContent = ReplayContent.ALL,
    ) -> Dict[str, ReplayStats]:
        """Write an ``incremental_dump`` delta back (see
        :meth:`Checkpointer.replay_increment`)."""
        return self._checkpointer.replay_increment(delta, content)
