import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
from torch.nn import Module

import pytorch_lightning as pl
from pytorch_lightning.overrides.base import _LightningModuleWrapperBase, unwrap_lightning_module
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin

from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities.enums import _StrategyType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _BAGUA_AVAILABLE
from pytorch_lightning.utilities.seed import reset_seed

if _BAGUA_AVAILABLE:
    import bagua.torch_api as bagua
    from bagua.torch_api.algorithms import Algorithm
    from bagua.torch_api.algorithms.q_adam import QAdamOptimizer
    from bagua.torch_api.communication import allreduce_inplace, barrier, broadcast_object, is_initialized, ReduceOp
    from bagua.torch_api.data_parallel.distributed import DistributedDataParallel_V1_9_0 as BaguaDistributedDataParallel


log = logging.getLogger(__name__)


class LightningBaguaModule(_LightningModuleWrapperBase):
    def __init__(self, pl_module: "pl.LightningModule") -> None:
        super().__init__(pl_module)
        # Bagua use `bagua_module_name` to distinguish different modules
        self._bagua_module_name = pl_module._get_name() + str(id(pl_module))

    def forward(self, *inputs, **kwargs):
        return super().forward(*inputs, **kwargs)


class BaguaStrategy(DDPStrategy):

    distributed_backend = _StrategyType.BAGUA

    def __init__(
        self,
        algorithm: str = "gradient_allreduce",
        gradient_as_bucket_view: bool = True,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ):

        super().__init__(
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

        self._bagua_algorithm = algorithm
        self._bagua_gradient_as_bucket_view = gradient_as_bucket_view
        self._bagua_kwargs = kwargs

    def setup_environment(self) -> None:
        # start the other scripts
        if not self.cluster_environment.creates_processes_externally:
            self._call_children_scripts()

        self.setup_distributed()

    def setup_distributed(self):
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        self._init_bagua_distributed()

    def setup(self, trainer: "pl.Trainer") -> None:
        super().setup(trainer)

        # move the model to the correct device
        self.model_to_device()

        model = LightningBaguaModule(self.model)
        self.configure_bagua_ddp(model)

    def _init_bagua_distributed(self):

        self._set_node_environment_variables()
        log.info(
            "initializing Bagua distributed: "
            f"GLOBAL_RANK: {self.global_rank}, "
            f"MEMBER: {self.global_rank + 1}/{self.world_size}"
        )

        # need to set device first before initialize Bagua distributed environment
        torch.cuda.set_device(self.local_rank)

        if not is_initialized():
            bagua.init_process_group()

    def _set_node_environment_variables(self) -> None:
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["NODE_RANK"] = str(self.node_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def _check_qadam_optimizer(self):

        trainer = self.lightning_module.trainer
        has_qadam_optimizer = any([isinstance(opt, QAdamOptimizer) for opt in trainer.optimizers])

        if not has_qadam_optimizer or len(trainer.optimizers) > 1 or len(trainer.lr_schedulers) > 1:
            raise MisconfigurationException("Bagua QAdam can only accept one QAdamOptimizer and one LR Scheduler.")

    def configure_bagua_ddp(self, model: Module):

        if self._bagua_algorithm == "qadam":
            self._check_qadam_optimizer()
            self._bagua_kwargs["q_adam_optimizer"] = self.optimizers[0]

        algorithm = Algorithm.init(self._bagua_algorithm, **self._bagua_kwargs)

        self._model = BaguaDistributedDataParallel(
            module=model,
            optimizers=self.optimizers,
            algorithm=algorithm,
            gradient_as_bucket_view=self._bagua_gradient_as_bucket_view,
        )

    def start_training(self, trainer: "pl.Trainer") -> Any:
        # start the background communication for async algorithm
        if self._bagua_algorithm == "async":
            self.model.bagua_algorithm.resume(self.model)

        return trainer.run_stage()

    def train_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        model = self._model
        if isinstance(model, BaguaDistributedDataParallel):
            model = model.module
        return unwrap_lightning_module(model) if model is not None else None

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register("bagua", cls, description="Default Bagua Plugin")

    def teardown(self) -> None:
        # abort the background communication for async algorithm, this operation is idempotent
        if self._bagua_algorithm == "async":
            self.model.bagua_algorithm.abort(self.model)

        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()

    def barrier(self, *args, **kwargs) -> None:
        if is_initialized():
            barrier()

    def broadcast(self, obj, src: int = 0) -> object:
        return broadcast_object(obj, src)

    def reduce(self, tensor, group: Optional[Any] = None, reduce_op: Union[ReduceOp, str] = "mean") -> torch.Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if group is not None:
            raise ValueError("Bagua does not support allreduce using a subcommunicator at this time. Unset `group`.")

        if isinstance(reduce_op, str):
            if reduce_op.lower() in ("avg", "mean"):
                op = ReduceOp.AVG
            elif reduce_op.lower() == "sum":
                op = ReduceOp.SUM
            else:
                raise ValueError(f"unrecognized `reduce_op`: {reduce_op}")

            allreduce_inplace(tensor, op=op)
            return tensor
