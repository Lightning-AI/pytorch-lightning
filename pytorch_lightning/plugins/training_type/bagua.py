import enum
import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module

import pytorch_lightning as pl
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities.enums import _StrategyType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _BAGUA_AVAILABLE
from pytorch_lightning.utilities.seed import reset_seed

if _BAGUA_AVAILABLE:
    import bagua.torch_api as bagua
    from bagua.torch_api.algorithms import Algorithm
    from bagua.torch_api.algorithms.q_adam import QAdamOptimizer
    from bagua.torch_api.data_parallel.distributed import DistributedDataParallel_V1_9_0 as BaguaDistributedDataParallel


log = logging.getLogger(__name__)


class BaguaPlugin(DDPPlugin):

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

        super(BaguaPlugin, self).__init__(
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

    def _init_bagua_distributed(self):

        self._set_node_environment_variables()
        log.info(
            "initializing Bagua distributed: "
            f"GLOBAL_RANK: {self.global_rank}, "
            f"MEMBER: {self.global_rank + 1}/{self.world_size}"
        )

        torch.cuda.set_device(self.local_rank)
        bagua.init_process_group()

    def _set_node_environment_variables(self) -> None:
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)
        os.environ["RANK"] = str(self.global_rank)
        os.environ["NODE_RANK"] = str(self.node_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)

    def pre_dispatch(self):
        # move the model to the correct device
        self.model_to_device()

        model = LightningDistributedModule(self.model)
        self.configure_bagua_ddp(model)

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
        if self._bagua_algorithm == "async":
            self.model.bagua_algorithm.resume(self.model)

        return trainer.run_stage()

    def post_dispatch(self, trainer: "pl.Trainer"):
        if self._bagua_algorithm == "async":
            self.model.bagua_algorithm.abort(self.model)

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        model = self._model
        if isinstance(model, BaguaDistributedDataParallel):
            model = model.module
        return unwrap_lightning_module(model) if model is not None else None

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register("bagua", cls, description="Default Bagua Plugin")
