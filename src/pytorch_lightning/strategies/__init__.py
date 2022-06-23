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
from pytorch_lightning.strategies.bagua import BaguaStrategy  # noqa: F401
from pytorch_lightning.strategies.ddp import DDPStrategy  # noqa: F401
from pytorch_lightning.strategies.ddp2 import DDP2Strategy  # noqa: F401
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy  # noqa: F401
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy  # noqa: F401
from pytorch_lightning.strategies.dp import DataParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.fully_sharded import DDPFullyShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.fully_sharded_native import DDPFullyShardedNativeStrategy  # noqa: F401
from pytorch_lightning.strategies.hivemind import HivemindStrategy  # noqa: F401
from pytorch_lightning.strategies.horovod import HorovodStrategy  # noqa: F401
from pytorch_lightning.strategies.hpu_parallel import HPUParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.ipu import IPUStrategy  # noqa: F401
from pytorch_lightning.strategies.parallel import ParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.sharded import DDPShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.sharded_spawn import DDPSpawnShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy  # noqa: F401
from pytorch_lightning.strategies.single_hpu import SingleHPUStrategy  # noqa: F401
from pytorch_lightning.strategies.single_tpu import SingleTPUStrategy  # noqa: F401
from pytorch_lightning.strategies.strategy import Strategy  # noqa: F401
from pytorch_lightning.strategies.strategy_registry import call_register_strategies, StrategyRegistry  # noqa: F401
from pytorch_lightning.strategies.tpu_spawn import TPUSpawnStrategy  # noqa: F401

STRATEGIES_BASE_MODULE = "pytorch_lightning.strategies"

call_register_strategies(STRATEGIES_BASE_MODULE)
