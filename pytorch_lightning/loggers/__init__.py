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
from os import environ

from pytorch_lightning.loggers.base import LightningLoggerBase, LoggerCollection
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

__all__ = [
    'LightningLoggerBase',
    'LoggerCollection',
    'TensorBoardLogger',
    'CSVLogger',
]

from pytorch_lightning.loggers.comet import _COMET_AVAILABLE, CometLogger  # noqa: F401
from pytorch_lightning.loggers.mlflow import _MLFLOW_AVAILABLE, MLFlowLogger  # noqa: F401
from pytorch_lightning.loggers.neptune import _NEPTUNE_AVAILABLE, NeptuneLogger  # noqa: F401
from pytorch_lightning.loggers.test_tube import _TESTTUBE_AVAILABLE, TestTubeLogger  # noqa: F401
from pytorch_lightning.loggers.wandb import _WANDB_AVAILABLE, WandbLogger  # noqa: F401

if _COMET_AVAILABLE:
    __all__.append('CometLogger')
    # needed to prevent ImportError and duplicated logs.
    environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

if _MLFLOW_AVAILABLE:
    __all__.append('MLFlowLogger')

if _NEPTUNE_AVAILABLE:
    __all__.append('NeptuneLogger')

if _TESTTUBE_AVAILABLE:
    __all__.append('TestTubeLogger')

if _WANDB_AVAILABLE:
    __all__.append('WandbLogger')
