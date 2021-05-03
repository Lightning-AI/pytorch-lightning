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
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.finetuning import BackboneFinetuning, BaseFinetuning
from pytorch_lightning.callbacks.gpu_stats_monitor import GPUStatsMonitor
from pytorch_lightning.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from pytorch_lightning.callbacks.lambda_function import LambdaCallback
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.prediction_writer import BasePredictionWriter
from pytorch_lightning.callbacks.progress import ProgressBar, ProgressBarBase
from pytorch_lightning.callbacks.pruning import ModelPruning
from pytorch_lightning.callbacks.quantization import QuantizationAwareTraining
from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from pytorch_lightning.callbacks.timer import Timer

__all__ = [
    'BackboneFinetuning',
    'BaseFinetuning',
    'Callback',
    'EarlyStopping',
    'GPUStatsMonitor',
    'GradientAccumulationScheduler',
    'LambdaCallback',
    'LearningRateMonitor',
    'ModelCheckpoint',
    'ModelPruning',
    'BasePredictionWriter',
    'ProgressBar',
    'ProgressBarBase',
    'QuantizationAwareTraining',
    'StochasticWeightAveraging',
    'Timer',
]
