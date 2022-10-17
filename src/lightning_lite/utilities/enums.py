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

from typing import TYPE_CHECKING

from lightning_utilities.core.enums import StrEnum

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


class _StrategyType(LightningEnum):
    """Define type of training strategy."""

    DP = "dp"
    DDP = "ddp"
    DDP_SPAWN = "ddp_spawn"
    DDP_FORK = "ddp_fork"
    DEEPSPEED = "deepspeed"
    DDP_SHARDED = "ddp_sharded"
    DDP_SHARDED_SPAWN = "ddp_sharded_spawn"
    DDP_FULLY_SHARDED = "ddp_fully_sharded"

    @staticmethod
    def interactive_compatible_types() -> list[_StrategyType]:
        """Returns a list containing interactive compatible _StrategyTypes."""
        return [
            _StrategyType.DP,
            _StrategyType.DDP_FORK,
        ]

    def is_interactive_compatible(self) -> bool:
        """Returns whether self is interactive compatible."""
        return self in _StrategyType.interactive_compatible_types()


class _AcceleratorType(LightningEnum):
    """Define Accelerator type by its nature."""

    CPU = "CPU"
    CUDA = "CUDA"
    TPU = "TPU"
    MPS = "MPS"
