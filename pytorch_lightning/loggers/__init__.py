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

try:
    # needed to prevent ImportError and duplicated logs.
    environ["COMET_DISABLE_AUTO_LOGGING"] = "1"

    from pytorch_lightning.loggers.comet import CometLogger
except ImportError:  # pragma: no-cover
    del environ["COMET_DISABLE_AUTO_LOGGING"]  # pragma: no-cover
else:
    __all__.append('CometLogger')

try:
    from pytorch_lightning.loggers.mlflow import MLFlowLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('MLFlowLogger')

try:
    from pytorch_lightning.loggers.neptune import NeptuneLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('NeptuneLogger')

try:
    from pytorch_lightning.loggers.test_tube import TestTubeLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('TestTubeLogger')

try:
    from pytorch_lightning.loggers.wandb import WandbLogger
except ImportError:  # pragma: no-cover
    pass  # pragma: no-cover
else:
    __all__.append('WandbLogger')
