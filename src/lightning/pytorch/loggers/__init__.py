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
import os

from lightning.pytorch.loggers.comet import _COMET_AVAILABLE, CometLogger  # noqa: F401
from lightning.pytorch.loggers.csv_logs import CSVLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.mlflow import _MLFLOW_AVAILABLE, MLFlowLogger  # noqa: F401
from lightning.pytorch.loggers.neptune import NeptuneLogger  # noqa: F401
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.loggers.wandb import WandbLogger  # noqa: F401

__all__ = ["CSVLogger", "Logger", "TensorBoardLogger"]

if _COMET_AVAILABLE:
    __all__.append("CometLogger")
    # needed to prevent ModuleNotFoundError and duplicated logs.
    os.environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

if _MLFLOW_AVAILABLE:
    __all__.append("MLFlowLogger")
