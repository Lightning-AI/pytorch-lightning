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

from pytorch_lightning.utilities import LightningEnum


class TrainerState(LightningEnum):
    """ State for the :class:`~pytorch_lightning.trainer.trainer.Trainer`
    to indicate what is currently or was executed. It follows the user-called
    functions such as `trainer.fit()` and `trainer.test().

    >>> # you can compare the type with a string
    >>> TrainerState.FITTING == 'FITTING'
    True
    >>> # which is case insensitive
    >>> TrainerState.FINISHED == 'finished'
    True
    """
    INITIALIZING = 'INITIALIZING'  # trainer creation
    FITTING = 'FITTING'  # trainer.fit()
    VALIDATING = 'VALIDATING'  # trainer.validate()
    TESTING = 'TESTING'  # trainer.test()
    PREDICTING = 'PREDICTING'  # trainer.predict()
    TUNING = 'TUNING'  # trainer.tune()
    FINISHED = 'FINISHED'
    INTERRUPTED = 'INTERRUPTED'

    @property
    def stopped(self) -> bool:
        return self in (self.FINISHED, self.INTERRUPTED)

    @property
    def running(self) -> bool:
        return self in (self.FITTING, self.VALIDATING, self.TESTING, self.PREDICTING, self.TUNING)


class RunningStage(LightningEnum):
    """Current running stage.

    This stage complements :class:`TrainerState` for example to indicate that
    `RunningStage.VALIDATING` will be set both during `TrainerState.FITTING`
    and `TrainerState.VALIDATING`. It follows the internal code logic.

    >>> # you can match the Enum with string
    >>> RunningStage.TRAINING == 'train'
    True
    """
    TRAINING = 'train'
    SANITY_CHECKING = 'sanity_check'
    VALIDATING = 'validation'
    TESTING = 'test'
    PREDICTING = 'predict'
    TUNING = 'tune'

    @property
    def evaluating(self) -> bool:
        return self in (self.VALIDATING, self.TESTING)
