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
from pytorch_lightning.strategies.executors.base import Executor
from pytorch_lightning.strategies.executors.ddp import DDPSubprocessExecutor
from pytorch_lightning.strategies.executors.ddp_spawn import DDPSpawnExecutor
from pytorch_lightning.strategies.executors.single_process import SingleProcessExecutor
from pytorch_lightning.strategies.executors.tpu_spawn import TPUSpawnExecutor

__all__ = [
    "DDPSpawnExecutor",
    "DDPSubprocessExecutor",
    "Executor",
    "SingleProcessExecutor",
    "TPUSpawnExecutor",
]
