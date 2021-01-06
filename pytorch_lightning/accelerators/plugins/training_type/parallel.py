from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import List, Optional
import torch
from pytorch_lightning.accelerators.plugins.training_type.training_type_plugin import TrainingTypePlugin
from pytorch_lightning.cluster_environments import ClusterEnvironment
from pytorch_lightning.core import LightningModule
from pytorch_lightning.overrides.data_parallel import LightningDistributedDataParallel

if torch.distributed.is_available():
    from torch.distributed import ReduceOp
else:

    class ReduceOp:
        SUM = None

class ParallelPlugin(TrainingTypePlugin, ABC):
    def __init__(
        self,
        parallel_devices: List[torch.device],
        cluster_environment: Optional[ClusterEnvironment] = None,
    ):
        super().__init__()
        self.parallel_devices = parallel_devices
        self.local_rank = 0
        self.world_size = 1
        self.cluster_environment = cluster_environment

    @property
    @abstractmethod
    def root_device(self):
        raise NotImplementedError

    @property
    def on_gpu(self):
        return self.root_device.type == "cuda" and torch.cuda.is_available()

    @abstractmethod
    def setup(self, model):
        raise NotImplementedError

    def connect(self, model, *args, **kwargs):
        self.setup(model)
        return self.model

    @property
    def is_global_zero(self) -> bool:
        return self.global_rank == 0

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(
            num_replicas=len(self.parallel_devices),
            rank=self.global_rank
        )
        return distributed_sampler_kwargs

    def reduce_early_stopping_decision(self, should_stop: bool) -> bool:
        should_stop = torch.tensor(int(should_stop), device=self.lightning_module.device)
        should_stop = self.reduce(should_stop, reduce_op=ReduceOp.SUM)
        should_stop = bool(should_stop == self.world_size)
        return should_stop

    @staticmethod
    def configure_sync_batchnorm(model: LightningModule) -> LightningModule:
        """
        Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override to synchronize batchnorm between specific process groups instead
        of the whole world or use a different sync_bn like `apex`'s version.

        Args:
            model: pointer to current :class:`LightningModule`.

        Return:
            LightningModule with batchnorm layers synchronized between process groups
        """
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        return model

    @contextmanager
    def block_backward_sync(self):
        """
        Blocks ddp sync gradients behaviour on backwards pass.
        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(self.model, LightningDistributedDataParallel):
            yield self.model.no_sync()
        else:
            yield None