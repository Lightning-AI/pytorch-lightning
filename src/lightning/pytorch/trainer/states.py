# Copyright The Lightning AI team.
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
from dataclasses import dataclass
from typing import Optional

from lightning.pytorch.utilities.enums import LightningEnum


class TrainerStatus(LightningEnum):
    """Enum for the status of the :class:`~lightning.pytorch.trainer.trainer.Trainer`"""

    INITIALIZING = "initializing"  # trainer creation
    RUNNING = "running"
    FINISHED = "finished"
    INTERRUPTED = "interrupted"

    @property
    def stopped(self) -> bool:
        return self in (self.FINISHED, self.INTERRUPTED)


class TrainerFn(LightningEnum):
    """Enum for the user-facing functions of the :class:`~lightning.pytorch.trainer.trainer.Trainer` such as
    :meth:`~lightning.pytorch.trainer.trainer.Trainer.fit` and
    :meth:`~lightning.pytorch.trainer.trainer.Trainer.test`."""

    FITTING = "fit"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"


class RunningStage(LightningEnum):
    """Enum for the current running stage.

    This stage complements :class:`TrainerFn` by specifying the current running stage for each function.
    More than one running stage value can be set while a :class:`TrainerFn` is running:

        - ``TrainerFn.FITTING`` - ``RunningStage.{SANITY_CHECKING,TRAINING,VALIDATING}``
        - ``TrainerFn.VALIDATING`` - ``RunningStage.VALIDATING``
        - ``TrainerFn.TESTING`` - ``RunningStage.TESTING``
        - ``TrainerFn.PREDICTING`` - ``RunningStage.PREDICTING``

    """

    TRAINING = "train"
    SANITY_CHECKING = "sanity_check"
    VALIDATING = "validate"
    TESTING = "test"
    PREDICTING = "predict"

    @property
    def evaluating(self) -> bool:
        return self in (self.VALIDATING, self.TESTING, self.SANITY_CHECKING)

    @property
    def dataloader_prefix(self) -> Optional[str]:
        if self in (self.VALIDATING, self.SANITY_CHECKING):
            return "val"
        return self.value


@dataclass
class TrainerState:
    """Dataclass to encapsulate the current :class:`~lightning.pytorch.trainer.trainer.Trainer` state."""

    status: TrainerStatus = TrainerStatus.INITIALIZING
    fn: Optional[TrainerFn] = None
    stage: Optional[RunningStage] = None

    @property
    def finished(self) -> bool:
        return self.status == TrainerStatus.FINISHED

    @property
    def stopped(self) -> bool:
        return self.status.stopped
