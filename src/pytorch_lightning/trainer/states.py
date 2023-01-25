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
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from typing import Any, List, Optional

from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.enums import _FaultTolerantMode
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


class _DeprecationManagingEnumMeta(EnumMeta):
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
        member: _DeprecationManagingEnumMeta = super().__getitem__(name)
        member.deprecate()
        return member

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        obj = super().__call__(*args, **kwargs)
        if isinstance(obj, Enum):
            obj.deprecate()
        return obj


class TrainerStatus(LightningEnum):
    """Enum for the status of the :class:`~pytorch_lightning.trainer.trainer.Trainer`"""

    INITIALIZING = "initializing"  # trainer creation
    RUNNING = "running"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"

    @property
    def stopped(self) -> bool:
        return self in (self.FINISHED, self.INTERRUPTED)


class TrainerFn(LightningEnum, metaclass=_DeprecationManagingEnumMeta):
    """
    Enum for the user-facing functions of the :class:`~pytorch_lightning.trainer.trainer.Trainer`
    such as :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit` and
    :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`.
    """

    FITTING = "fit"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"
    TUNING = "tune"

    def deprecate(self) -> None:
        if self == self.TUNING:
            rank_zero_deprecation(
                f"`TrainerFn.{self.name}` has been deprecated in v1.8.0 and will be removed in v2.0.0."
            )

    @classmethod
    def _without_tune(cls) -> List["TrainerFn"]:
        fns = [fn for fn in cls if fn != "tune"]
        return fns


class RunningStage(LightningEnum, metaclass=_DeprecationManagingEnumMeta):
    """Enum for the current running stage.

    This stage complements :class:`TrainerFn` by specifying the current running stage for each function.
    More than one running stage value can be set while a :class:`TrainerFn` is running:

        - ``TrainerFn.FITTING`` - ``RunningStage.{SANITY_CHECKING,TRAINING,VALIDATING}``
        - ``TrainerFn.VALIDATING`` - ``RunningStage.VALIDATING``
        - ``TrainerFn.TESTING`` - ``RunningStage.TESTING``
        - ``TrainerFn.PREDICTING`` - ``RunningStage.PREDICTING``
        - ``TrainerFn.TUNING`` - ``RunningStage.{TUNING,SANITY_CHECKING,TRAINING,VALIDATING}``
    """

    TRAINING = "train"
    SANITY_CHECKING = "sanity_check"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"
    TUNING = "tune"

    @property
    def evaluating(self) -> bool:
        return self in (self.VALIDATING, self.TESTING)

    @property
    def dataloader_prefix(self) -> Optional[str]:
        if self == self.SANITY_CHECKING:
            return None
        if self == self.VALIDATING:
            return "val"
        return self.value

    def deprecate(self) -> None:
        if self == self.TUNING:
            rank_zero_deprecation(
                f"`RunningStage.{self.name}` has been deprecated in v1.8.0 and will be removed in v2.0.0."
            )

    @classmethod
    def _without_tune(cls) -> List["RunningStage"]:
        fns = [fn for fn in cls if fn != "tune"]
        return fns


@dataclass
class TrainerState:
    """Dataclass to encapsulate the current :class:`~pytorch_lightning.trainer.trainer.Trainer` state."""

    status: TrainerStatus = TrainerStatus.INITIALIZING
    fn: Optional[TrainerFn] = None
    stage: Optional[RunningStage] = None

    # detect the fault tolerant flag
    _fault_tolerant_mode: _FaultTolerantMode = field(default_factory=_FaultTolerantMode.detect_current_mode)

    @property
    def finished(self) -> bool:
        return self.status == TrainerStatus.FINISHED

    @property
    def stopped(self) -> bool:
        return self.status.stopped
