import logging
import os
import enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.nn import Module

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.environments.bagua_environment import BaguaEnvironment
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.utilities.enums import _StrategyType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.overrides.base import unwrap_lightning_module
from pytorch_lightning.utilities.imports import _BAGUA_AVAILABLE

if _BAGUA_AVAILABLE:
    import bagua.torch_api as bagua
    from bagua.torch_api.data_parallel.distributed import DistributedDataParallel_V1_9_0 as DistributedDataParallel


log = logging.getLogger(__name__)


class BaguaDistributedAlgorithm(enum.Enum):
    GradientAllReduce = "gradient_allreduce"
    ByteGrad = "bytegrad"
    Decentralized = "decentralized"
    LowPrecisionDecentralized = "low_prec_decentralized"
    QAdam = "qadam"
    AsyncModelAverage = "async"

    @staticmethod
    def from_str(val: str):
        if not isinstance(val, str):
            raise ValueError("BaguaDistributedAlgorithm name must be a string, but got: {}".format(val))

        reverse_dict = {e.value: e for e in BaguaDistributedAlgorithm}
        return reverse_dict[val]


class BaguaPlugin(DDPPlugin):

    distributed_backend = _StrategyType.BAGUA

    def __init__(
        self,
        algorithm: Union[BaguaDistributedAlgorithm, str] = BaguaDistributedAlgorithm.GradientAllReduce,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        gradient_as_bucket_view: bool = True,
        **kwargs: Union[Any, Dict[str, Any]],
    ):

        super(BaguaPlugin, self).__init__(
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )

        if isinstance(algorithm, str):
            algorithm = BaguaDistributedAlgorithm.from_str(algorithm)

        self._bagua_algorithm = algorithm
        self._bagua_gradient_as_bucket_view = gradient_as_bucket_view
        self._bagua_kwargs = kwargs
        if isinstance(cluster_environment, BaguaEnvironment):
            self._bagua_service_port = cluster_environment.service_port
        else:
            self._bagua_service_port = None

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
        # TODO
        if self._bagua_service_port is None:
            self._bagua_service_port = 23574

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
        os.environ["BAGUA_SERVICE_PORT"] = str(self._bagua_service_port)

    def pre_dispatch(self):
        # move the model to the correct device
        self.model_to_device()

        model = LightningDistributedModule(self.model)
        self.configure_bagua_ddp(model)

    def configure_bagua_ddp(self, model: Module):

        # TODO: format kwargs
        if self._bagua_algorithm == BaguaDistributedAlgorithm.GradientAllReduce:
            from bagua.torch_api.algorithms.gradient_allreduce import GradientAllReduceAlgorithm

            algorithm = GradientAllReduceAlgorithm(**self._bagua_kwargs)
        elif self._bagua_algorithm == BaguaDistributedAlgorithm.ByteGrad:
            from bagua.torch_api.algorithms.bytegrad import ByteGradAlgorithm

            algorithm = ByteGradAlgorithm(**self._bagua_kwargs)
        elif self._bagua_algorithm == BaguaDistributedAlgorithm.Decentralized:
            from bagua.torch_api.algorithms.decentralized import DecentralizedAlgorithm

            algorithm = DecentralizedAlgorithm(**self._bagua_kwargs)
        elif self._bagua_algorithm == BaguaDistributedAlgorithm.LowPrecisionDecentralized:
            from bagua.torch_api.algorithms.decentralized import LowPrecisionDecentralizedAlgorithm

            algorithm = LowPrecisionDecentralizedAlgorithm(**self._bagua_kwargs)
        elif self._bagua_algorithm == BaguaDistributedAlgorithm.QAdam:
            from bagua.torch_api.algorithms.q_adam import QAdamAlgorithm

            algorithm = QAdamAlgorithm(**self._bagua_kwargs)
        elif self._bagua_algorithm == BaguaDistributedAlgorithm.AsyncModelAverage:
            from bagua.torch_api.algorithms.async_model_average import AsyncModelAverageAlgorithm

            algorithm = AsyncModelAverageAlgorithm(**self._bagua_kwargs)
        else:
            raise MisconfigurationException("Unsupport Bagua algorithm.")

        self._model = DistributedDataParallel(
            module=model,
            optimizers=self.optimizers,
            algorithm=algorithm,
            gradient_as_bucket_view=self._bagua_gradient_as_bucket_view,
        )

    def start_training(self, trainer: "pl.Trainer") -> Any:
        if self._bagua_algorithm == BaguaDistributedAlgorithm.AsyncModelAverage:
            self.model.bagua_algorithm.resume(self.model)

        return trainer.run_stage()

    def post_dispatch(self, trainer: "pl.Trainer"):
        if self._bagua_algorithm == BaguaDistributedAlgorithm.AsyncModelAverage:
            self.model.bagua_algorithm.abort(self.model)

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        model = self._model
        if isinstance(model, DistributedDataParallel):
            model = model.module
        return unwrap_lightning_module(model) if model is not None else None

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register("bagua", cls, description="Default Bagua Plugin")
