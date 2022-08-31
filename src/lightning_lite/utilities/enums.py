from __future__ import annotations

from enum import Enum


class LightningEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases."""

    @classmethod
    def from_str(cls, value: str) -> LightningEnum | None:
        statuses = cls.__members__.keys()
        for st in statuses:
            if st.lower() == value.lower():
                return cls[st]
        return None

    def __eq__(self, other: object) -> bool:
        other = other.value if isinstance(other, Enum) else str(other)
        return self.value.lower() == other.lower()

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.value.lower())


class AMPType(LightningEnum):
    """Type of Automatic Mixed Precission used for training.

    >>> # you can match the type with string
    >>> AMPType.APEX == 'apex'
    True
    """

    APEX = "apex"
    NATIVE = "native"


class PrecisionType(LightningEnum):
    """Type of precision used.

    >>> PrecisionType.HALF == 16
    True
    >>> PrecisionType.HALF in (16, "16")
    True
    """

    HALF = "16"
    FLOAT = "32"
    FULL = "64"
    BFLOAT = "bf16"
    MIXED = "mixed"

    @staticmethod
    def supported_type(precision: str | int) -> bool:
        return any(x == precision for x in PrecisionType)

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in PrecisionType]


class _StrategyType(LightningEnum):
    """Define type of training strategy.

    >>> # you can match the type with string
    >>> _StrategyType.DDP == 'DDP'
    True
    >>> # which is case invariant
    >>> _StrategyType.DP in ('dp', )
    True
    """

    DP = "dp"
    DDP = "ddp"
    DDP_SPAWN = "ddp_spawn"
    DDP_FORK = "ddp_fork"
    TPU_SPAWN = "tpu_spawn"
    DEEPSPEED = "deepspeed"
    HOROVOD = "horovod"
    DDP_SHARDED = "ddp_sharded"
    DDP_SHARDED_SPAWN = "ddp_sharded_spawn"
    DDP_FULLY_SHARDED = "ddp_fully_sharded"
    BAGUA = "bagua"
    HPU_PARALLEL = "hpu_parallel"

    @staticmethod
    def interactive_compatible_types() -> list[_StrategyType]:
        """Returns a list containing interactive compatible _StrategyTypes."""
        return [
            _StrategyType.DP,
            _StrategyType.TPU_SPAWN,
            _StrategyType.DDP_FORK,
        ]

    def is_interactive_compatible(self) -> bool:
        """Returns whether self is interactive compatible."""
        return self in _StrategyType.interactive_compatible_types()


class _AcceleratorType(LightningEnum):
    """Define Accelerator type by its nature.

    >>> _AcceleratorType.CPU == _AcceleratorType.from_str('cpu')
    True
    >>> # you can match the type with string
    >>> _AcceleratorType.CUDA == 'CUDA'
    True
    >>> # which is case invariant
    >>> _AcceleratorType.TPU in ('tpu', 'CPU')
    True
    """

    CPU = "CPU"
    CUDA = "CUDA"
    IPU = "IPU"
    TPU = "TPU"
    HPU = "HPU"
    MPS = "MPS"
