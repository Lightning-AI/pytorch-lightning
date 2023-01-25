# Copyright The Lightning team.
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
from enum import Enum, EnumMeta
from typing import Any

from lightning_fabric.utilities.enums import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class _DeprecatedEnumMeta(EnumMeta):
    """Enum that calls `deprecate()` whenever a member is accessed.

    Adapted from: https://stackoverflow.com/a/62309159/208880
    """

    def __getattribute__(cls, name: str) -> Any:
        obj = super().__getattribute__(name)
        # ignore __dunder__ names -- prevents potential recursion errors
        if not (name.startswith("__") and name.endswith("__")) and isinstance(obj, Enum):
            obj.deprecate()
        return obj

    def __getitem__(cls, name: str) -> Any:
        member: _DeprecatedEnumMeta = super().__getitem__(name)
        member.deprecate()
        return member

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        obj = super().__call__(*args, **kwargs)
        if isinstance(obj, Enum):
            obj.deprecate()
        return obj


class PrecisionType(LightningEnum, metaclass=_DeprecatedEnumMeta):
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

    def deprecate(self) -> None:
        rank_zero_deprecation(
            f"The `{type(self).__name__}` enum has been deprecated in v1.9.0 and will be removed in v2.0.0."
            f" Use the string value `{self.value!r}` instead."
        )


class AMPType(LightningEnum, metaclass=_DeprecatedEnumMeta):
    """Type of Automatic Mixed Precision used for training."""

    APEX = "apex"
    NATIVE = "native"

    def deprecate(self) -> None:
        rank_zero_deprecation(
            f"The `{type(self).__name__}` enum has been deprecated in v1.9.0 and will be removed in v2.0.0."
            f" Use the string value `{self.value!r}` instead."
        )


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
