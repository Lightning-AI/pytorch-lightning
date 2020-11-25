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

from pytorch_lightning.loggers.comet import COMET_AVAILABLE, CometLogger
from pytorch_lightning.loggers.mlflow import MLFLOW_AVAILABLE, MLFlowLogger
from pytorch_lightning.loggers.neptune import NEPTUNE_AVAILABLE, NeptuneLogger
from pytorch_lightning.loggers.test_tube import TESTTUBE_AVAILABLE, TestTubeLogger
from pytorch_lightning.loggers.wandb import WANDB_AVAILABLE, WandbLogger

if COMET_AVAILABLE:
    __all__.append('CometLogger')
    # needed to prevent ImportError and duplicated logs.
    environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

if MLFLOW_AVAILABLE:
    __all__.append('MLFlowLogger')

if NEPTUNE_AVAILABLE:
    __all__.append('NeptuneLogger')

if TESTTUBE_AVAILABLE:
    __all__.append('TestTubeLogger')

if WANDB_AVAILABLE:
    __all__.append('WandbLogger')
