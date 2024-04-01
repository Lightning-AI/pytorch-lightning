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
from lightning_pytorch.callbacks.batch_size_finder import BatchSizeFinder
from lightning_pytorch.callbacks.callback import Callback
from lightning_pytorch.callbacks.checkpoint import Checkpoint
from lightning_pytorch.callbacks.device_stats_monitor import DeviceStatsMonitor
from lightning_pytorch.callbacks.early_stopping import EarlyStopping
from lightning_pytorch.callbacks.finetuning import BackboneFinetuning, BaseFinetuning
from lightning_pytorch.callbacks.gradient_accumulation_scheduler import GradientAccumulationScheduler
from lightning_pytorch.callbacks.lambda_function import LambdaCallback
from lightning_pytorch.callbacks.lr_finder import LearningRateFinder
from lightning_pytorch.callbacks.lr_monitor import LearningRateMonitor
from lightning_pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning_pytorch.callbacks.model_summary import ModelSummary
from lightning_pytorch.callbacks.on_exception_checkpoint import OnExceptionCheckpoint
from lightning_pytorch.callbacks.prediction_writer import BasePredictionWriter
from lightning_pytorch.callbacks.progress import ProgressBar, RichProgressBar, TQDMProgressBar
from lightning_pytorch.callbacks.pruning import ModelPruning
from lightning_pytorch.callbacks.rich_model_summary import RichModelSummary
from lightning_pytorch.callbacks.spike import SpikeDetection
from lightning_pytorch.callbacks.stochastic_weight_avg import StochasticWeightAveraging
from lightning_pytorch.callbacks.throughput_monitor import ThroughputMonitor
from lightning_pytorch.callbacks.timer import Timer

__all__ = [
    "BackboneFinetuning",
    "BaseFinetuning",
    "BasePredictionWriter",
    "BatchSizeFinder",
    "Callback",
    "Checkpoint",
    "DeviceStatsMonitor",
    "EarlyStopping",
    "GradientAccumulationScheduler",
    "LambdaCallback",
    "LearningRateFinder",
    "LearningRateMonitor",
    "ModelCheckpoint",
    "ModelPruning",
    "ModelSummary",
    "OnExceptionCheckpoint",
    "ProgressBar",
    "RichModelSummary",
    "RichProgressBar",
    "StochasticWeightAveraging",
    "SpikeDetection",
    "ThroughputMonitor",
    "Timer",
    "TQDMProgressBar",
]
