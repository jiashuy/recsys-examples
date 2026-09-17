import abc
from dataclasses import replace
from typing import Dict, List, Optional

from dynamicemb.dynamicemb_config import *
from dynamicemb.types import (
    DEFAULT_UNIFORM_LOWER,
    DEFAULT_UNIFORM_UPPER,
    group_key_of,
)
from dynamicemb_extensions import (
    CurandStateContext,
    const_init,
    const_init_per_table,
    debug_init,
    normal_init,
    normal_init_per_table,
    truncated_normal_init,
    truncated_normal_init_per_table,
    uniform_init,
    uniform_init_per_table,
)


def _with_default_bounds(
    args: DynamicEmbInitializerArgs,
) -> DynamicEmbInitializerArgs:
    """Bounds are the only parameters that may still be unset this late.

    The planner fills them from a table's num_embeddings, and nothing else has
    to. Returns a copy rather than filling them in place: the args belong to
    the caller's configuration, which stays as it was written.
    """
    if args.lower is not None and args.upper is not None:
        return args
    return replace(
        args,
        lower=DEFAULT_UNIFORM_LOWER if args.lower is None else args.lower,
        upper=DEFAULT_UNIFORM_UPPER if args.upper is None else args.upper,
    )


class MultiTableInitializer(abc.ABC):
    """The initializer of a fused module: one object covering all its tables.

    A fused module keeps several logical tables in one value buffer. They are
    grouped by initializer *mode* only -- see
    :meth:`~dynamicemb.dynamicemb_config.DynamicEmbTableOptions.get_grouped_key`
    -- so the mode is common but the parameters need not be, and routinely are
    not: an unbounded UNIFORM defaults to ``+/-sqrt(1 / num_embeddings)``, which
    is per table.

    When the tables do agree, or there is only one, each subclass calls its
    mode's plain kernel with the parameters as scalars, exactly as a
    single-table initializer would. When they do not, it calls that mode's
    ``_per_table`` kernel with a ``[num_tables, num_params]`` tensor, and the
    kernel looks up the parameters of the table owning each row.
    """

    @staticmethod
    def create(
        args_list: List[DynamicEmbInitializerArgs],
        device: torch.device,
    ) -> "MultiTableInitializer":
        """The initializer for a module whose tables are configured like this."""
        if not args_list:
            raise ValueError("A module needs at least one table to initialize.")
        # Ask what the tables were grouped on rather than compare modes here,
        # so this cannot drift from what actually decided they may be fused.
        keys = {group_key_of(args) for args in args_list}
        if len(keys) != 1:
            raise ValueError(
                f"Tables of one module must agree on their initializer, got "
                f"{len(keys)} different ones: {keys}"
            )
        mode = args_list[0].mode
        if mode not in _INITIALIZERS:
            raise ValueError(f"Not supported initializer type: {mode}")

        args_list = [_with_default_bounds(args) for args in args_list]
        initializer_class = _INITIALIZERS[mode]
        rows = initializer_class.table_parameters(args_list)
        table_parameters = None
        if any(row != rows[0] for row in rows):
            table_parameters = torch.tensor(
                rows, dtype=torch.float32, device=device
            )
        return initializer_class(args_list[0], table_parameters)

    def __init__(
        self,
        args: DynamicEmbInitializerArgs,
        table_parameters: Optional[torch.Tensor] = None,
    ):
        self._args = args
        self._table_parameters = table_parameters

    @staticmethod
    def table_parameters(
        args_list: List[DynamicEmbInitializerArgs],
    ) -> List[List[float]]:
        """Each table's parameters, in the order this mode's kernel reads them."""
        return [[] for _ in args_list]

    def _table_ids_for_kernel(self, table_ids: Optional[torch.Tensor]):
        if table_ids is None:
            raise ValueError(
                "This module's tables initialize differently, so the initializer "
                "needs the table id of every row it writes."
            )
        return table_ids

    @abc.abstractmethod
    def __call__(
        self,
        buffer: torch.Tensor,
        indices: torch.Tensor,
        keys: Optional[torch.Tensor],  # remove it when debug mode is removed
        table_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize ``buffer[indices]``.

        ``keys`` and ``table_ids`` are addressed by buffer row, not by position
        within ``indices`` -- the convention the kernels use for both.
        """
        ...


class NormalInitializer(MultiTableInitializer):
    def __init__(self, args, table_parameters=None):
        super().__init__(args, table_parameters)
        self._curand_state = CurandStateContext()

    @staticmethod
    def table_parameters(args_list):
        return [[args.mean, args.std_dev] for args in args_list]

    def __call__(self, buffer, indices, keys, table_ids=None) -> None:
        if self._table_parameters is None:
            normal_init(
                buffer,
                indices,
                self._curand_state,
                self._args.mean,
                self._args.std_dev,
            )
        else:
            normal_init_per_table(
                buffer,
                indices,
                self._curand_state,
                self._table_parameters,
                self._table_ids_for_kernel(table_ids),
            )


class TruncatedNormalInitializer(MultiTableInitializer):
    def __init__(self, args, table_parameters=None):
        super().__init__(args, table_parameters)
        self._curand_state = CurandStateContext()

    @staticmethod
    def table_parameters(args_list):
        return [
            [args.mean, args.std_dev, args.lower, args.upper] for args in args_list
        ]

    def __call__(self, buffer, indices, keys, table_ids=None) -> None:
        if self._table_parameters is None:
            truncated_normal_init(
                buffer,
                indices,
                self._curand_state,
                self._args.mean,
                self._args.std_dev,
                self._args.lower,
                self._args.upper,
            )
        else:
            truncated_normal_init_per_table(
                buffer,
                indices,
                self._curand_state,
                self._table_parameters,
                self._table_ids_for_kernel(table_ids),
            )


class UniformInitializer(MultiTableInitializer):
    def __init__(self, args, table_parameters=None):
        super().__init__(args, table_parameters)
        self._curand_state = CurandStateContext()

    @staticmethod
    def table_parameters(args_list):
        return [[args.lower, args.upper] for args in args_list]

    def __call__(self, buffer, indices, keys, table_ids=None) -> None:
        if self._table_parameters is None:
            uniform_init(
                buffer,
                indices,
                self._curand_state,
                self._args.lower,
                self._args.upper,
            )
        else:
            uniform_init_per_table(
                buffer,
                indices,
                self._curand_state,
                self._table_parameters,
                self._table_ids_for_kernel(table_ids),
            )


class ConstantInitializer(MultiTableInitializer):
    @staticmethod
    def table_parameters(args_list):
        return [[args.value] for args in args_list]

    def __call__(self, buffer, indices, keys, table_ids=None) -> None:
        if self._table_parameters is None:
            const_init(buffer, indices, self._args.value)
        else:
            const_init_per_table(
                buffer,
                indices,
                self._table_parameters,
                self._table_ids_for_kernel(table_ids),
            )


class DebugInitializer(MultiTableInitializer):
    # Fills a row from its key, so it takes no parameters at all and its tables
    # cannot disagree. Inherits table_parameters, which then reports no
    # parameters for every table, so create never builds a tensor for it.
    def __call__(self, buffer, indices, keys, table_ids=None) -> None:
        debug_init(buffer, indices, keys)


_INITIALIZERS: Dict[DynamicEmbInitializerMode, type] = {
    DynamicEmbInitializerMode.NORMAL: NormalInitializer,
    DynamicEmbInitializerMode.TRUNCATED_NORMAL: TruncatedNormalInitializer,
    DynamicEmbInitializerMode.UNIFORM: UniformInitializer,
    DynamicEmbInitializerMode.CONSTANT: ConstantInitializer,
    DynamicEmbInitializerMode.DEBUG: DebugInitializer,
}
