from pathlib import Path

from pytorch_lightning.strategies.ddp import DDPStrategy  # noqa: F401
from pytorch_lightning.strategies.ddp2 import DDP2Strategy  # noqa: F401
from pytorch_lightning.strategies.ddp_spawn import DDPSpawnStrategy  # noqa: F401
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy  # noqa: F401
from pytorch_lightning.strategies.dp import DataParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.fully_sharded import DDPFullyShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.horovod import HorovodStrategy  # noqa: F401
from pytorch_lightning.strategies.ipu import IPUStrategy  # noqa: F401
from pytorch_lightning.strategies.parallel import ParallelStrategy  # noqa: F401
from pytorch_lightning.strategies.sharded import DDPShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.sharded_spawn import DDPSpawnShardedStrategy  # noqa: F401
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy  # noqa: F401
from pytorch_lightning.strategies.single_tpu import SingleTPUStrategy  # noqa: F401
from pytorch_lightning.strategies.strategy import Strategy  # noqa: F401
from pytorch_lightning.strategies.strategy_registry import call_register_strategies, StrategyRegistry  # noqa: F401
from pytorch_lightning.strategies.tpu_spawn import TPUSpawnStrategy  # noqa: F401

FILE_ROOT = Path(__file__).parent
STRATEGIES_BASE_MODULE = "pytorch_lightning.strategies"

call_register_strategies(FILE_ROOT, STRATEGIES_BASE_MODULE)
