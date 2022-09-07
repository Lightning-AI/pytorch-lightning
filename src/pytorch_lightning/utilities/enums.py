# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Enumerated utilities."""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

from lightning_utilities.core.enums import StrEnum

from pytorch_lightning.utilities.exceptions import MisconfigurationException

if TYPE_CHECKING:
    from enum import Enum

    # re-defined because `mypy` infers `StrEnum` as `Any`
    class LightningEnum(StrEnum, Enum):
        ...

else:
    LightningEnum = StrEnum


class AMPType(LightningEnum):
    """Type of Automatic Mixed Precission used for training."""

    APEX = "apex"
    NATIVE = "native"


class PrecisionType(LightningEnum):
    """Type of precision used."""

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


class GradClipAlgorithmType(LightningEnum):
    """Define gradient_clip_algorithm types - training-tricks.
    NORM type means "clipping gradients by norm". This computed over all model parameters together.
    VALUE type means "clipping gradients by value". This will clip the gradient value for each parameter.

    References:
        clip_by_norm: https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_norm_
        clip_by_value: https://pytorch.org/docs/stable/nn.html#torch.nn.utils.clip_grad_value_
    """

    VALUE = "value"
    NORM = "norm"

    @staticmethod
    def supported_type(val: str) -> bool:
        return any(x.value == val for x in GradClipAlgorithmType)

    @staticmethod
    def supported_types() -> list[str]:
        return [x.value for x in GradClipAlgorithmType]


class AutoRestartBatchKeys(LightningEnum):
    """Defines special dictionary keys used to track captured dataset state with multiple workers."""

    PL_RESTART_META = "__pl_restart_meta"


class _StrategyType(LightningEnum):
    """Define type of training strategy."""

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
    """Define Accelerator type by its nature."""

    CPU = "CPU"
    CUDA = "CUDA"
    IPU = "IPU"
    TPU = "TPU"
    HPU = "HPU"
    MPS = "MPS"


class _FaultTolerantMode(LightningEnum):

    DISABLED = "disabled"
    AUTOMATIC = "automatic"
    MANUAL = "manual"

    @property
    def is_enabled(self) -> bool:
        return self is not _FaultTolerantMode.DISABLED

    @property
    def is_automatic(self) -> bool:
        return self is _FaultTolerantMode.AUTOMATIC

    @property
    def is_manual(self) -> bool:
        return self is _FaultTolerantMode.MANUAL

    @classmethod
    def detect_current_mode(cls) -> _FaultTolerantMode:
        """This classmethod detects if `Fault Tolerant` is activated and maps its value to `_FaultTolerantMode`."""
        env_value = os.getenv("PL_FAULT_TOLERANT_TRAINING", "0").lower()
        # the int values are kept for backwards compatibility, but long-term we want to keep only the strings
        if env_value in ("0", "disabled"):
            return _FaultTolerantMode.DISABLED
        elif env_value in ("1", "automatic"):
            return _FaultTolerantMode.AUTOMATIC
        elif env_value in ("2", "manual"):
            return _FaultTolerantMode.MANUAL
        raise MisconfigurationException(
            "The environment flag `PL_FAULT_TOLERANT_TRAINING` should be either 'disabled', 'automatic', or 'manual'."
        )
