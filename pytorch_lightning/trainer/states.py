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
from dataclasses import dataclass
from typing import Optional

from pytorch_lightning.utilities import LightningEnum


class TrainerStatus(LightningEnum):
    """Status of the :class:`~pytorch_lightning.trainer.trainer.Trainer`"""
    INITIALIZING = 'initializing'  # trainer creation
    RUNNING = 'running'
    FINISHED = 'finished'
    INTERRUPTED = 'interrupted'

    @property
    def stopped(self) -> bool:
        return self in (self.FINISHED, self.INTERRUPTED)


class TrainerFn(LightningEnum):
    """
    Functions of the :class:`~pytorch_lightning.trainer.trainer.Trainer`
    functions such as ``trainer.fit()`` and ``trainer.test()``.
    """
    FITTING = 'fit'  # trainer.fit()
    VALIDATING = 'validate'  # trainer.validate()
    TESTING = 'test'  # trainer.test()
    PREDICTING = 'predict'  # trainer.predict()
    TUNING = 'tune'  # trainer.tune()

    @property
    def _setup_fn(self) -> 'TrainerFn':
        """
        'fit' is used instead of 'tune' as there aren't 'tune_dataloaders'.

        This is used for the ``setup()`` and ``teardown()`` hooks
        """
        return TrainerFn.FITTING if self == TrainerFn.TUNING else self


class RunningStage(LightningEnum):
    """Current running stage.

    This stage complements :class:`TrainerState`, for example, to indicate that
    `RunningStage.VALIDATING` will be set both during `TrainerFn.FITTING`
    and `TrainerFn.VALIDATING`. It follows the internal code logic.
    """
    TRAINING = 'train'
    SANITY_CHECKING = 'sanity_check'
    VALIDATING = 'validate'
    TESTING = 'test'
    PREDICTING = 'predict'
    TUNING = 'tune'

    @property
    def evaluating(self) -> bool:
        return self in (self.VALIDATING, self.TESTING)


@dataclass
class TrainerState:
    """Dataclass to encapsulate the current :class:`~pytorch_lightning.trainer.trainer.Trainer` state"""
    status: TrainerStatus = TrainerStatus.INITIALIZING
    fn: Optional[TrainerFn] = None
    stage: Optional[RunningStage] = None

    @property
    def finished(self) -> bool:
        return self.status == TrainerStatus.FINISHED

    @property
    def stopped(self) -> bool:
        return self.status.stopped
