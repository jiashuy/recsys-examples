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

"""The checkpoint of a module: files, incremental dump, replay.
"""

import json
import os
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch  # usort:skip
import torch.distributed as dist
from dynamicemb.checkpoint.paths import (
    encode_checkpoint_file_path,
    encode_counter_checkpoint_file_path,
    encode_meta_json_file_path,
    get_loading_files,
)
from dynamicemb.dynamicemb_config import (
    DynamicEmbTableOptions,
    EvictedItemMode,
    ReplayContent,
    score_strategy_has_timestamp_column,
)
from dynamicemb.key_ownership import owned_key_mask, owned_keys
from dynamicemb.storage.cache import flush_cache
from dynamicemb.storage.chained_storage import ChainedHybridStorage, ChainedStorage
from dynamicemb.types import Cache, ReplayStats
from dynamicemb_extensions import device_timestamp
from torch import Tensor  # usort:skip

if TYPE_CHECKING:
    from dynamicemb.incremental_dump import DeltaDumpResult


class ShardInfo:
    """Which shard of its tables this rank holds, and the group that shares them.

    Everything that names a shard -- checkpoint files, the load-time key filter, the delta
    meta -- derives from this rather than from the global process group.
    """

    def __init__(self, shard_ranks: Optional[List[int]], table_names: List[str]) -> None:
        global_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.ranks: List[int] = (
            list(shard_ranks) if shard_ranks is not None else list(range(world_size))
        )
        if global_rank not in self.ranks:
            raise ValueError(
                f"Rank {global_rank} holds no shard of tables {table_names} "
                f"(shards on ranks {self.ranks})."
            )
        self.num_shards = len(self.ranks)
        self.index = self.ranks.index(global_rank)
        self.sharding_type = (
            "row_wise" if self.num_shards == world_size else "table_row_wise"
        )
        # Set through set_group by the table-row-wise sharding; a barrier over the world
        # would deadlock when nodes hold different table counts.
        self.pg: Optional[dist.ProcessGroup] = None
        self._table_names = table_names

    def set_group(self, pg: dist.ProcessGroup) -> None:
        """Name the group of ranks that hold this kernel's tables.

        Collectives over the kernel -- the checkpoint barrier, the incremental
        dump's gather -- run over the shards of a table, not over the world. By
        default that is the world, which is right for row-wise; table-row-wise
        calls this with the intra-node group, because the other node is busy
        with its own tables and may have a different count of them.

        The group must be the one the plan described: ``shard_ranks`` decides
        who owns which key and how the checkpoint is named, and this group
        decides who waits for whom. They are built by different code -- the
        planner and the sharding -- and this is the one place they meet, so the
        sizes are checked against each other here.
        """
        size = dist.get_world_size(group=pg)
        if size != self.num_shards:
            raise ValueError(
                f"Shard group of {size} ranks does not match the "
                f"{self.num_shards} shards the plan gave tables "
                f"{self._table_names} (shards on ranks {self.ranks})."
            )
        self.pg = pg

    def barrier(self, pg: Optional[dist.ProcessGroup] = None) -> None:
        """Wait for the ranks that hold this kernel's tables.

        The group the sharding named through ``set_group`` wins over the one the
        caller offered, because a caller holds a single group for a whole model
        while each kernel's tables may sit on a different node -- under
        table-row-wise no one group is right for all of them. ``pg`` is what is
        left to go on when the sharding named nothing: row-wise, where a table is
        on every rank and the caller's group is the right one.
        """
        if not dist.is_initialized():
            return
        dist.barrier(group=self.pg if self.pg is not None else pg)

    def write_meta(self, meta_file_path: str) -> None:
        with open(meta_file_path) as meta_file:
            meta_data = json.load(meta_file)
        meta_data["sharding_type"] = self.sharding_type
        meta_data["num_shards"] = self.num_shards
        meta_data["shard_ranks"] = self.ranks
        with open(meta_file_path, "w") as meta_file:
            json.dump(meta_data, meta_file, indent=4)


@dataclass
class _ReplayJob:
    """One table's share of a delta, resolved and checked, ready to write.

    Produced by the planning pass and consumed by the applying one, so that "no
    table is written until every table has been checked" is a property of the
    control flow rather than something to be read out of a long loop.
    """

    table_id: int  # where this table sits among the module's own
    name: str
    keys: Tensor
    values: Tensor
    optimizer_states: Optional[Tensor]
    scores: Optional[Tensor]
    slot_index: Tensor
    # Only ``erased_keys`` is ever replayed as a removal. ``evicted_keys`` is
    # not one a replica has to perform -- the key that took the evicted one's
    # slot is in this very delta and overwrites it -- but the cache is a second
    # index that write does not reach, so it is carried for invalidation.
    erased_keys: Optional[Tensor]
    evicted_keys: Optional[Tensor]
    # The source's score bookkeeping, adopted before writing so a restored key
    # lands on the scale the replica will later threshold against.
    current_score: Optional[int]


class Checkpointer:
    """Writes and reads a module's tables, and the incremental dump and replay.

    Holds what a checkpoint touches -- the storage, the cache in front of it, the
    admission counter, the per-table scores -- and the shard this rank holds. The
    module's ``dump``, ``load``, ``incremental_dump`` and ``replay_increment`` call it.
    """

    def __init__(
        self,
        storage: Union[ChainedStorage, ChainedHybridStorage],
        cache: Optional["Cache"],
        flush: Callable[[], None],
        table_names: List[str],
        options: List["DynamicEmbTableOptions"],
        shard: ShardInfo,
        scores: Dict[str, int],
        admission_counter: Optional[Any],
    ) -> None:
        self._storage = storage
        self._cache = cache
        self._flush = flush
        self._table_names = table_names
        self._dynamicemb_options = options
        self._shard = shard
        self._scores = scores
        self._admission_counter = admission_counter

    def dump(
        self,
        save_dir: str,
        optim: bool = False,
        counter: bool = False,
        table_names: Optional[List[str]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ) -> None:
        if table_names is None:
            table_names = self._table_names

        # Files are named by the table's own shard, not by the caller's rank in
        # pg: a table-row-wise table has local_world_size shards on one node.
        rank = self._shard.index
        world_size = self._shard.num_shards

        self._flush()

        counter_table = self._admission_counter
        for table_id, table_name in enumerate(self._table_names):
            if table_name not in set(table_names):
                continue

            meta_file_path = encode_meta_json_file_path(save_dir, table_name)
            current_score = (
                self._scores.get(table_name, None) if hasattr(self, "_scores") else None
            )

            emb_key_path = encode_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "keys"
            )
            emb_value_path = encode_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "values"
            )
            emb_score_path = encode_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "scores"
            )
            opt_value_path = encode_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "opt_values"
            )

            storage = self._storage
            self._shard.barrier(pg)
            ts = device_timestamp()
            storage.dump(
                table_id,
                meta_file_path,
                emb_key_path,
                emb_value_path,
                emb_score_path,
                opt_value_path,
                include_optim=optim,
                include_meta=(rank == 0),
                current_score=current_score,
                timestamp=ts,
            )
            if rank == 0:
                self._shard.write_meta(meta_file_path)

            # No counter means no counter-based admission (e.g. probabilistic
            # admission), so there is nothing to dump.
            if not counter or counter_table is None:
                continue

            counter_key_path = encode_counter_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "keys"
            )
            counter_frequency_path = encode_counter_checkpoint_file_path(
                save_dir, table_name, rank, world_size, "frequencies"
            )
            counter_table.dump(counter_key_path, counter_frequency_path, table_id)

    def load(
        self,
        save_dir: str,
        optim: bool = False,
        counter: bool = False,
        table_names: Optional[List[str]] = None,
        pg: Optional[dist.ProcessGroup] = None,
    ):
        if table_names is None:
            table_names = self._table_names

        # Files are looked up by the table's own shard, as dump named them:
        # not by the caller's rank in pg, which pg does not decide either way.
        rank = self._shard.index
        world_size = self._shard.num_shards

        storage = self._storage
        counter_table = self._admission_counter
        # get_loading_files raises the missing-directory error itself.
        sorted_file_names = (
            sorted(os.listdir(save_dir)) if os.path.isdir(save_dir) else None
        )
        for table_id, table_name in enumerate(self._table_names):
            if table_name not in set(table_names):
                continue

            meta_json_file = encode_meta_json_file_path(save_dir, table_name)

            (
                emb_key_files,
                emb_value_files,
                emb_score_files,
                opt_value_files,
                counter_key_files,
                counter_frequency_files,
            ) = get_loading_files(
                save_dir,
                table_name,
                rank=rank,
                world_size=world_size,
                sorted_file_names=sorted_file_names,
            )
            if len(emb_key_files) == 0:
                continue

            num_key_files = len(emb_key_files)
            self._shard.barrier(pg)
            ts = device_timestamp()
            for i in range(num_key_files):
                loaded_score = storage.load(
                    table_id,
                    meta_json_file,
                    emb_key_files[i],
                    emb_value_files[i],
                    emb_score_files[i] if len(emb_score_files) > 0 else None,
                    opt_value_files[i] if len(opt_value_files) > 0 else None,
                    include_optim=optim,
                    timestamp=ts,
                    shard_index=rank,
                    num_shards=world_size,
                )
                if loaded_score is not None and table_name in self._scores:
                    self._scores[table_name] = loaded_score

            if not counter:
                continue
            if counter_table is None:
                # Only worth a warning when the checkpoint has counter state to drop.
                if counter_key_files:
                    warnings.warn(
                        f"Counter table is none and will not load for table: {table_name}"
                    )
                continue
            num_counter_key_files = len(counter_key_files)
            for i in range(num_counter_key_files):
                counter_table.load(
                    counter_key_files[i],
                    counter_frequency_files[i],
                    table_id,
                    self._dynamicemb_options[table_id].dist_type,
                    shard_index=rank,
                    num_shards=world_size,
                )

    def export_keys_values(
        self, table_name: str, device: torch.device, batch_size: int = 65536
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return this rank's (keys, embeddings) for one table.

        Embeddings come back at the table's own precision, the same as they are
        checkpointed and the same as ``incremental_dump`` reports them.
        """
        self._flush()

        table_id = self._table_names.index(table_name)
        keys_list = []
        values_list = []

        for keys, embeddings, _, _ in self._storage.export_keys_values(
            device, batch_size, table_id
        ):
            keys_list.append(keys)
            values_list.append(embeddings)

        if len(keys_list) == 0:
            return torch.empty(0, dtype=torch.int64, device=device), torch.empty(
                0, 0, dtype=self._storage.embedding_dtype(), device=device
            )
        return torch.cat(keys_list), torch.cat(values_list, dim=0)

    def pop_evicted_keys(
        self, table_names: Optional[List[str]] = None
    ) -> Dict[str, Tensor]:
        """Return + clear this rank's retained evicted keys, per table.

        Only tables configured with ``evicted_item_mode=RETAIN_KEY`` are
        included; others are omitted. Returns
        ``{table_name: 1-D unique keys}`` on host. The keys are this rank's local
        shard only (row-wise sharded, so disjoint across ranks); cross-rank
        aggregation is the model-level ``pop_evicted_keys``'s job via ``pg``.
        Clearing affects only this rank.

        Keys removed by an explicit erase are **not** here -- see
        :meth:`pop_erased_keys`.
        """
        return self._pop_retained(
            "pop_evicted_keys",
            lambda mode: EvictedItemMode.RETAIN_KEY in mode,
            table_names,
        )

    def pop_erased_keys(
        self, table_names: Optional[List[str]] = None
    ) -> Dict[str, Tensor]:
        """Return + clear the keys an explicit erase removed, per table.

        The counterpart of :meth:`pop_evicted_keys`, kept apart from it because
        the two mean different things to a consumer: an eviction is reproduced by
        whoever takes over the slot, while an erase leaves the slot to nobody and
        has to be replayed as a removal.

        Every table is included, unlike :meth:`pop_evicted_keys`: whether an
        erase was recorded is that ``erase`` call's decision, not the table's, so
        there is no configuration to filter on -- a table nobody asked to record
        simply returns an empty tensor.
        """
        return self._pop_retained("pop_erased_keys", lambda mode: True, table_names)

    def _pop_retained(
        self,
        method: str,
        wanted: Callable[[EvictedItemMode], bool],
        table_names: Optional[List[str]],
    ) -> Dict[str, Tensor]:
        """Drain one of the storage's retained-key buffers, per table."""
        storage = self._storage
        if not hasattr(storage, method):
            return {}
        pop = getattr(storage, method)
        result: Dict[str, Tensor] = {}
        for i, name in enumerate(self._table_names):
            if not wanted(self._dynamicemb_options[i].evicted_item_mode):
                continue
            if table_names is not None and name not in table_names:
                continue
            result[name] = pop(i).cpu()  # host tensor
        return result

    def incremental_dump(
        self,
        named_thresholds: Dict[str, int],
        pg: Optional[dist.ProcessGroup] = None,
    ) -> "DeltaDumpResult":
        """Dump keys/values (+ evicted keys + meta) whose score crosses the threshold.

        Returns a :class:`DeltaDumpResult` for this module (column-aligned lists by
        table). ``values[i]`` is embeddings only, with the rest of each stored
        row in ``optimizer_states[i]`` and every score word in ``scores[i]``;
        ``meta[i]`` carries current_score / slot_index / current_capacity /
        row_capacity / bucket_capacity / num_scores / world_size /
        table_options; ``evicted_keys[i]`` is the keys this table
        retained since the last dump -- evictions and explicit erases under
        ``RETAIN_KEY`` -- and ``erased_keys[i]`` the keys an explicit erase
        asked to have recorded. Both are drained here.

        The meaning of the threshold depends on the table's score strategy:

        - Strategies that carry a device-timestamp column (single ``TIMESTAMP``, or a
          compound strategy that includes ``TIMESTAMP`` such as ``(TIMESTAMP, LFU)``)
          produce a *time-based* incremental dump: only items whose last-access
          timestamp crosses the threshold (i.e. touched/changed since the reference
          time) are dumped. Use the value returned by :meth:`get_score` as the
          reference threshold.
        - Strategies without a timestamp column (e.g. ``STEP``, single ``LFU``,
          ``NO_EVICTION``) instead threshold on the absolute score value: items whose
          score is not less than the threshold are dumped. This is not a time-based
          increment.
        """
        from dynamicemb.incremental_dump import (  # lazy: avoid import cycle
            DeltaDumpResult,
        )

        # A table-row-wise kernel exists only on its shard group's ranks, so a
        # gather over a wider group would wait on ranks that never call it.
        if pg is not None and self._shard.pg is not None:
            pg = self._shard.pg

        storage = self._storage
        if not isinstance(storage, (ChainedStorage, ChainedHybridStorage)):
            raise TypeError(
                f"incremental_dump requires a chained storage, "
                f"got {type(storage).__name__}"
            )
        if self._cache is not None and isinstance(
            storage, ChainedStorage
        ):
            flush_cache(self._cache, storage)
        res = DeltaDumpResult()
        # One reference timestamp for the whole call: the age baseline for the
        # dumped timestamp score columns, and the returned current_score.
        # Sampled BEFORE the dump so a key touched *during* the dump falls into
        # the next window (at-least-once) instead of being missed.
        ts = device_timestamp()
        for table_name, threshold in named_thresholds.items():
            if table_name not in self._table_names:
                warnings.warn(
                    f"incremental_dump: table_name '{table_name}' is not in this "
                    f"module (available: {self._table_names}); skipping.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            table_id = self._table_names.index(table_name)
            (
                keys_cat,
                values_cat,
                slot_index,
                opt_states,
                scores,
            ) = storage.incremental_dump(table_id, threshold, pg, timestamp=ts)
            option = self._dynamicemb_options[table_id]
            if score_strategy_has_timestamp_column(option.score_strategy):
                current_score = ts
            else:
                current_score = self._scores[table_name]
            # Drain both retained-key buffers, each gated by what its table
            # actually retains, and aggregate within the SAME pg as keys/values.
            ev = self._drain_retained(
                storage,
                "pop_evicted_keys",
                EvictedItemMode.RETAIN_KEY in option.evicted_item_mode,
                table_id,
                pg,
            )
            # Always drained: recording an erase is the erase call's decision,
            # so there is no table setting to gate on here.
            er = self._drain_retained(storage, "pop_erased_keys", True, table_id, pg)
            res.table_names.append(table_name)
            res.keys.append(keys_cat)
            res.values.append(values_cat)
            res.optimizer_states.append(opt_states)
            res.scores.append(scores)
            res.evicted_keys.append(ev)
            res.erased_keys.append(er)
            res.meta.append(
                {
                    "current_score": current_score,
                    "slot_index": slot_index,
                    "current_capacity": self._capacity_of(storage, table_id),
                    "row_capacity": self._row_capacity_of(storage, table_id),
                    "bucket_capacity": self._bucket_capacity_of(storage),
                    "num_scores": self._num_scores_of(storage),
                    "world_size": self._shard.num_shards,
                    "shard_index": self._shard.index,
                    "shard_ranks": list(self._shard.ranks),
                    "table_options": option,
                }
            )
        return res

    def _replay_compatibility(
        self, table_id: int, meta: Dict[str, Any]
    ) -> Optional[str]:
        """Why this table cannot take a slot-for-slot replay, or ``None``.

        A key is only ever probed inside its own home bucket, and that bucket is
        ``hash(key) % table_capacity / bucket_capacity`` -- so a source slot is
        only meaningful when the target's capacity and bucket layout match the
        source's. Everything compared here comes from the delta's ``meta``.

        ``score_strategy`` is compared **as configured**, tuple order included.
        A compound strategy's physical layout is fixed -- ``(TIMESTAMP, LFU)``
        and ``(LFU, TIMESTAMP)`` both store the timestamp at word 0 -- so the two
        agree on where a slot write lands. They do not agree on the delta: the
        configured order is the column order of the score block a dump emits,
        and replay reads that block with the *target's* order, deciding which
        columns are ages to rebase and how to permute them (:class:`ScoreRules`) by
        logical position. Two
        tables matching physically and differing logically would rebase the
        frequency as though it were a timestamp and write each word into the
        other's slot, with nothing to signal it. Requiring the configured order
        to match is what keeps that from being expressible.
        """
        storage = self._storage
        option = self._dynamicemb_options[table_id]
        src_options = meta.get("table_options")
        if src_options is None:
            return "delta carries no 'table_options' (dumped by an older version)"

        # A table places keys one by one, in whatever slot and row it hands out, so
        # nothing about slots, rows or buckets has to match.
        checks = [
            ("num_scores", meta.get("num_scores"), self._num_scores_of(storage)),
            ("world_size", meta.get("world_size"), self._shard.num_shards),
            # Configured order, not physical layout -- see the note above.
            (
                "score_strategy",
                src_options.score_strategy,
                option.score_strategy,
            ),
            ("dim", src_options.dim, option.dim),
            ("dist_type", src_options.dist_type, option.dist_type),
        ]
        for name, src, dst in checks:
            if src is None:
                return f"delta carries no '{name}' (dumped by an older version)"
            if src != dst:
                return f"{name} mismatch (source {src} vs target {dst})"
        return None

    def replay_increment(
        self,
        delta: "DeltaDumpResult",
        content: ReplayContent = ReplayContent.ALL,
    ) -> Dict[str, ReplayStats]:
        """Write an ``incremental_dump`` delta back into this module's tables.

        Replay is exact in *layout*: every key is written at the slot and value
        row it held in the source table, so the target ends up layout-identical
        to it. Only a table that matches the delta's metadata is replayed; one
        that does not is rejected with a ``ValueError`` rather than written some
        other way -- see :meth:`_replay_compatibility`.

        The key and its embedding always travel; *content* selects what comes
        along, and defaults to both, i.e. the replica ends up holding what the
        source held. Dropping ``SCORE`` leaves restored keys scored as if freshly
        inserted here, so the replica orders its own future evictions by when it
        received each key rather than by how the source ranked it; dropping
        ``OPTIMIZER_STATE`` keeps the state of a key already on its target row
        and initialises any other. See :class:`ReplayContent`.

        ``delta.erased_keys`` is applied whenever it holds anything: those keys
        were explicitly removed at the source and nothing else will remove them
        here. ``delta.evicted_keys`` is never applied -- writing the delta's
        slots already reproduces an eviction, because the key that took the
        evicted one's slot is in the delta. A table retaining evictions for some
        other consumer therefore costs a replay nothing.

        The target must be sharded across the same number of ranks as the
        source; a differing ``world_size`` is rejected, because a slot names a
        position inside one rank's table and two source ranks folded onto one
        target rank would collide on it. Within that fixed world, only the keys
        this rank owns are replayed, with ownership recomputed from the key. A
        delta gathered over a process group therefore fans out correctly when
        handed to every rank; a per-rank delta (``incremental_dump`` with
        ``pg=None``) should be replayed on the rank that produced it, or the
        filter drops all of it -- visible as ``ReplayStats.skipped``.

        Args:
            delta: one collection's :class:`DeltaDumpResult`. Tables not present
                in this module are skipped with a warning.

        Returns:
            ``{table_name: ReplayStats}`` -- keys written / removed / skipped
            per table.

        Raises:
            ValueError: a table's metadata does not match the source's, or the
                delta is missing the per-key data replay needs. Raised before
                this module writes anything.
            TypeError: this module's storage is not a chained storage.
            NotImplementedError: a table is sharded with
                ``dist_type="continuous"`` across more than one rank.
            RuntimeError: a key could not be written at its source slot even
                though the metadata matched. Unlike the checks above this fires
                mid-write, so the table may hold a partial replay.
        """
        storage = self._storage
        if not isinstance(storage, (ChainedStorage, ChainedHybridStorage)):
            raise TypeError(
                f"replay_increment requires a chained storage, "
                f"got {type(storage).__name__}"
            )
        # Two passes on purpose. A delta usually spans a whole collection, so
        # anything checked only when its table's turn came would raise after the
        # earlier tables had been erased from and written to -- a partly applied
        # delta, worse to recover from than a rejected one, and not what this
        # promises. Planning touches nothing; applying validates nothing.
        plan = self._plan_replay(delta, content, storage)

        if self._cache is not None and isinstance(
            storage, ChainedStorage
        ):
            # After planning, so a rejected delta leaves the model exactly as it
            # was. Pushing dirty cache entries down first makes the storage copy
            # this replay is about to overwrite the authoritative one.
            flush_cache(self._cache, storage)
            # Flushing can grow and rehash the backing table. Revalidate the
            # source slots after that layout-changing side effect, before any
            # replay writes use the original plan.
            plan = self._plan_replay(delta, content, storage)

        ts = device_timestamp()
        return {job.name: self._apply_replay(job, content, storage, ts) for job in plan}

    def _plan_replay(
        self,
        delta: "DeltaDumpResult",
        content: ReplayContent,
        storage: Union[ChainedStorage, ChainedHybridStorage],
    ) -> List["_ReplayJob"]:
        """Resolve and check every table in *delta*, writing nothing.

        Raises on the first table that cannot be replayed, so a caller that sees
        an exception knows *this module* is untouched. Not the collection: a
        collection whose tables do not all group together is several modules, and
        ``dynamicemb.replay_increment`` applies them one at a time.
        """
        plan: List[_ReplayJob] = []
        for i, table_name in enumerate(delta.table_names):
            if table_name not in self._table_names:
                warnings.warn(
                    f"replay_increment: table_name '{table_name}' is not in this "
                    f"module (available: {self._table_names}); skipping.",
                    UserWarning,
                    stacklevel=3,
                )
                continue
            table_id = self._table_names.index(table_name)
            meta = delta.meta[i]

            slot_index = meta.get("slot_index")
            if slot_index is None:
                raise ValueError(
                    f"replay_increment: delta for table '{table_name}' carries no "
                    "slot_index; it was produced by an incompatible version of "
                    "incremental_dump."
                )
            job = _ReplayJob(
                table_id=table_id,
                name=table_name,
                keys=delta.keys[i],
                values=delta.values[i],
                optimizer_states=(
                    delta.optimizer_states[i]
                    if ReplayContent.OPTIMIZER_STATE in content
                    else None
                ),
                scores=delta.scores[i] if ReplayContent.SCORE in content else None,
                slot_index=slot_index,
                erased_keys=delta.erased_keys[i],
                evicted_keys=delta.evicted_keys[i],
                current_score=meta.get("current_score"),
            )
            self._check_delta_shapes(job, content, storage)

            mismatch = self._replay_compatibility(table_id, meta)
            if mismatch is not None:
                raise ValueError(
                    f"replay_increment: cannot replay table '{table_name}' -- "
                    f"{mismatch}. Replay writes every key back at the slot it "
                    "held in the source table, which is only meaningful when the "
                    "two tables share a layout; configure the target to match the "
                    "source, or rebuild it from a full checkpoint instead."
                )
            plan.append(job)
        return plan

    def _check_delta_shapes(
        self,
        job: "_ReplayJob",
        content: ReplayContent,
        storage: Union[ChainedStorage, ChainedHybridStorage],
    ) -> None:
        """Every shape the write path depends on, checked before it runs.

        The storage layer checks the widths again -- it is callable on its own,
        without this module's planning pass -- but by then a table's predecessors
        have been written, so a malformed column on the last table of a
        collection would leave the rest of it advanced.
        """
        option = self._dynamicemb_options[job.table_id]
        n = job.keys.numel()

        rows = {"values": job.values.size(0), "slot_index": job.slot_index.numel()}
        if job.optimizer_states is not None:
            rows["optimizer_states"] = job.optimizer_states.size(0)
        if job.scores is not None:
            rows["scores"] = job.scores.size(0)
        if any(r != n for r in rows.values()):
            raise ValueError(
                f"replay_increment: delta columns for table '{job.name}' are not "
                f"row-aligned (keys={n}, "
                + ", ".join(f"{k}={v}" for k, v in rows.items())
                + ")."
            )

        if job.values.dim() != 2 or job.values.size(1) != option.dim:
            raise ValueError(
                f"replay_increment: table '{job.name}' has dim {option.dim}, but "
                f"the delta's embeddings have shape {tuple(job.values.shape)}."
            )

        if job.scores is not None:
            num_scores = self._num_scores_of(storage)
            # A single-word block may arrive 1-D; the storage layer reshapes it,
            # so accept the same two spellings here.
            width = job.scores.size(1) if job.scores.dim() == 2 else 1
            if job.scores.dim() > 2 or width != num_scores:
                raise ValueError(
                    f"replay_increment: table '{job.name}' has {num_scores} score "
                    f"word(s) per key, but the delta carries a score block of "
                    f"shape {tuple(job.scores.shape)}."
                )

        if ReplayContent.OPTIMIZER_STATE in content:
            optimizer = storage.tables[0].optimizer
            # Only meaningful where the table keeps per-row state at all; one
            # without it ignores the column rather than rejecting it.
            if optimizer.get_state_dim(option.dim) > 0:
                ckpt_dim = optimizer.get_ckpt_state_dim(option.dim)
                if job.optimizer_states is None:
                    # A delta dumped from a table whose optimizer keeps no state
                    # -- SGD into rowwise Adagrad, say. Nothing in the layout
                    # check catches it, and the write path raises, so without
                    # this the collection's earlier tables would already be in.
                    raise ValueError(
                        f"replay_increment: table '{job.name}' keeps optimizer "
                        "state per row, but the delta carries none. Drop "
                        "ReplayContent.OPTIMIZER_STATE, or replay a delta "
                        "dumped from a table with the same optimizer."
                    )
                if (
                    job.optimizer_states.dim() != 2
                    or job.optimizer_states.size(1) != ckpt_dim
                ):
                    raise ValueError(
                        f"replay_increment: table '{job.name}' dumps {ckpt_dim} "
                        f"optimizer-state column(s) per row, but the delta "
                        f"carries a block of shape "
                        f"{tuple(job.optimizer_states.shape)}."
                    )

    def _apply_replay(
        self,
        job: "_ReplayJob",
        content: ReplayContent,
        storage: Union[ChainedStorage, ChainedHybridStorage],
        ts: int,
    ) -> ReplayStats:
        """Write one planned table. Everything here has already been checked."""
        option = self._dynamicemb_options[job.table_id]
        stats = ReplayStats()
        keys, values = job.keys, job.values
        opt_states, scores, slot_index = (
            job.optimizer_states,
            job.scores,
            job.slot_index,
        )

        # An empty or absent list is normal -- nothing was erased, or rank
        # filtering left this rank none of them.
        erased, evicted = job.erased_keys, job.evicted_keys

        # Ownership follows how the table was sharded: this rank's index inside
        # the table's shard group, modulo the group's size -- the whole world
        # for row-wise, one node for table-row-wise. ``meta["world_size"]`` is
        # checked against that size. There is deliberately no process-group
        # argument: replay is local (filter by ownership, then write), so a
        # group could only narrow the modulus and mis-route every key, leaving
        # each claimed by several ranks and the one that owns it claiming nothing.
        rank = self._shard.index
        mask = owned_key_mask(keys, rank, self._shard.num_shards, option.dist_type)
        if mask is not None:
            stats.skipped = int(keys.numel() - mask.sum().item())
            keys, values = keys[mask], values[mask]
            if opt_states is not None:
                opt_states = opt_states[mask]
            if scores is not None:
                scores = scores[mask]
            slot_index = slot_index[mask]
            ws, dt = self._shard.num_shards, option.dist_type
            erased = owned_keys(erased, rank, ws, dt)
            evicted = owned_keys(evicted, rank, ws, dt)

        # Erase first: a key erased and then re-inserted inside the same window
        # appears in BOTH lists, and must survive the replay.
        if erased is not None and erased.numel() > 0:
            # Not recorded into this model's own erased buffer: these removals
            # came from upstream, and re-reporting them would make a chained
            # replica replay what it already received. A model that is itself a
            # dump source for someone further down would want the opposite --
            # say so when that case turns up.
            stats.erased = storage.erase_keys(job.table_id, erased)
        if (
            evicted is not None
            and evicted.numel() > 0
        ):
            # Writing slots does not reproduce an eviction here, so the keys the source
            # evicted are removed; one that was evicted and came back is in the delta,
            # and the write below puts it back.
            storage.erase_keys(job.table_id, evicted)

        # Before writing, so a restored key lands on the same scale the replica
        # will later threshold its own incremental_dump against. Timestamp-based
        # tables read their score off the device clock, so there is nothing to
        # carry -- their restored keys are stamped with ``ts``.
        if (
            job.current_score is not None
            and not score_strategy_has_timestamp_column(option.score_strategy)
            and job.name in self._scores
        ):
            self._scores[job.name] = job.current_score

        stats.merge(
            storage.replay_increment(
                job.table_id,
                keys,
                values,
                opt_states,
                scores,
                slot_index,
                self._scores.get(job.name, 0),
                content=content,
                timestamp=ts,
            )
        )
        self._invalidate_cache(job.table_id, keys, erased, evicted)
        return stats

    def _invalidate_cache(
        self,
        table_id: int,
        keys: Tensor,
        erased: Optional[Tensor],
        evicted: Optional[Tensor],
    ) -> None:
        """Drop from the cache everything the storage just stopped holding.

        The cache is a second index that writing a slot does not reach, so three
        groups have to go explicitly:

        - upserted keys, so the next lookup sees the value just written into the
          storage rather than the cached one;
        - erased keys, whose cached copy would resurrect them;
        - evicted keys, which lost their slot to a delta key. Replay ignores
          those as removals precisely because the write reproduces the eviction
          *in the storage* -- but a cached copy survives that, and
          ``flush_cache`` would write it back down on the next dump, undoing the
          eviction.

        A source table that does not retain evictions (``DISCARD``) cannot report
        that last group, so a caching replica that has to converge exactly wants
        ``evicted_item_mode=RETAIN_KEY``.
        """
        if self._cache is None:
            return
        parts = [keys]
        for extra in (erased, evicted):
            if extra is not None and extra.numel() > 0:
                parts.append(extra.to(keys.dtype))
        stale = torch.cat(parts) if len(parts) > 1 else keys
        if stale.numel() == 0:
            return
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._cache.invalidate_keys(stale.to(device=device), table_id)

    @staticmethod
    def _drain_retained(
        storage: Union[ChainedStorage, ChainedHybridStorage],
        method: str,
        wanted: bool,
        table_id: int,
        pg: Optional[dist.ProcessGroup],
    ) -> Optional[Tensor]:
        """Drain one retained-key buffer for one table, gathered and on host.

        ``None`` -- rather than an empty tensor -- when the table does not retain
        this kind of key at all, so a consumer can tell "nothing was removed"
        from "this table does not record removals".
        """
        from dynamicemb.incremental_dump import (  # lazy: avoid import cycle
            _all_gather_evicted_keys,
        )

        if not wanted or not hasattr(storage, method):
            return None
        out = getattr(storage, method)(table_id)
        if pg is not None:
            out = _all_gather_evicted_keys(out, pg)
        return out.cpu()

    @staticmethod
    def _row_capacity_of(
        storage: Union[ChainedStorage, ChainedHybridStorage], table_id: int
    ) -> Tuple[int, ...]:
        """Value-buffer rows per tier for one logical table.

        A second bound, distinct from :meth:`_capacity_of`. That one is the key
        map's capacity, which is the modulus for choosing a home bucket and so
        decides whether a *slot* means the same thing in two tables. This one is
        how many rows the value buffer actually has, which is what a *row* write
        is bounded by.

        The two coincide everywhere except NO_EVICTION, where the key map is
        deliberately ``1 / max_load_factor`` times the value buffer -- and
        rounding that up to a whole number of buckets makes the map's capacity
        non-injective in the buffer's. With ``bucket_capacity=128``, an
        ``init_capacity`` of 100 and of 128 both give a 256-slot key map while
        leaving 100 and 128 rows: equal on :meth:`_capacity_of`, and a source
        row of 127 written past the end of a 100-row target. Per tier rather
        than summed, since ``slot_index`` routes each key to one tier and a
        matching total would say nothing about either.

        Note the two meanings of ``tables`` in play: a storage's tiers, each of
        which has its own ``values``, a buffer per logical table.
        """
        return tuple(s.values.rows(table_id) for s in storage.tables)

    @staticmethod
    def _capacity_of(
        storage: Union[ChainedStorage, ChainedHybridStorage], table_id: int
    ) -> int:
        """Slots one logical table has across the whole storage.

        Summed over the tiers, because capacity is the one property here that a
        second tier adds to rather than duplicates -- the others below read a
        single tier.
        """
        return sum(s.key_index_map.capacity(table_id) for s in storage.tables)

    @staticmethod
    def _bucket_capacity_of(storage: Union[ChainedStorage, ChainedHybridStorage]) -> int:
        """The storage's hash-bucket capacity (HBM tier for a hybrid storage)."""
        return storage.tables[0].key_index_map.bucket_capacity_

    @staticmethod
    def _num_scores_of(storage: Union[ChainedStorage, ChainedHybridStorage]) -> int:
        """Score words per key (HBM tier for a hybrid storage)."""
        return storage.tables[0].key_index_map.num_scores_
