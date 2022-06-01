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
import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import torch.distributed
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.distributed import prepare_for_backward
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.launchers.spawn import _SpawnLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import (
    _get_process_group_backend_from_env,
    distributed_available,
    get_default_process_group_backend_for_device,
)
from pytorch_lightning.utilities.distributed import group as _group
from pytorch_lightning.utilities.distributed import (
    init_dist_connection,
    ReduceOp,
    register_ddp_comm_hook,
    sync_ddp_if_available,
)
from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_11
from pytorch_lightning.utilities.optimizer import optimizers_to_device
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import STEP_OUTPUT

log = logging.getLogger(__name__)


class DDPSpawnStrategy(ParallelStrategy):
    """Spawns processes using the :func:`torch.multiprocessing.spawn` method and joins processes after training
    finishes."""

    strategy_name = "ddp_spawn"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[callable] = None,
        ddp_comm_wrapper: Optional[callable] = None,
        process_group_backend: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self._num_nodes = 1
        self._ddp_kwargs = kwargs
        self._ddp_comm_state = ddp_comm_state
        self._ddp_comm_hook = ddp_comm_hook
        self._ddp_comm_wrapper = ddp_comm_wrapper
        self._local_rank = 0
        self._process_group_backend: Optional[str] = process_group_backend

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def root_device(self):
        return self.parallel_devices[self.local_rank]

    @property
    def num_processes(self):
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def _is_single_process_single_device(self):
        return True

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    def _configure_launcher(self):
        self._launcher = _SpawnLauncher(self)

    def setup(self, trainer: "pl.Trainer") -> None:
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)

        self.accelerator.setup(trainer)

        # move the model to the correct device
        self.model_to_device()

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            if self._layer_sync:
                self.model = self._layer_sync.apply(self.model)

        self.setup_precision_plugin()

        if trainer_fn == TrainerFn.FITTING:
            self.configure_ddp()

    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        return DistributedDataParallel(module=model, device_ids=self.determine_ddp_device_ids(), **self._ddp_kwargs)

    def set_world_ranks(self, process_idx: int = 0) -> None:
        self._local_rank = process_idx
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

    def _worker_setup(self, process_idx: int):
        reset_seed()
        self.set_world_ranks(process_idx)
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        init_dist_connection(self.cluster_environment, self._process_group_backend, self.global_rank, self.world_size)

    def _get_process_group_backend(self) -> str:
        return (
            self._process_group_backend
            or _get_process_group_backend_from_env()
            or get_default_process_group_backend_for_device(self.root_device)
        )

    def pre_configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)

    def _register_ddp_hooks(self) -> None:
        # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
        # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
        if self.root_device.type == "cuda" and self._is_single_process_single_device:
            register_ddp_comm_hook(
                model=self.model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )

    def configure_ddp(self) -> None:
        self.pre_configure_ddp()
        self.model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()

        # set up optimizers after the wrapped module has been moved to the device
        self.setup_optimizers(self.lightning_module.trainer)
        optimizers_to_device(self.optimizers, self.root_device)

    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def barrier(self, *args, **kwargs) -> None:
        if not distributed_available():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        if not distributed_available():
            return obj
        obj = [obj]
        if self.global_rank != src:
            obj = [None]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def model_to_device(self):
        if self.root_device.type == "cuda":
            # set the device on the spawned subprocesses
            torch.cuda.set_device(self.root_device)
        self.model.to(self.root_device)

    def pre_backward(self, closure_loss: torch.Tensor) -> None:
        """Run before precision plugin executes backward."""
        if not self.lightning_module.automatic_optimization:
            prepare_for_backward(self.model, closure_loss)

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
        if isinstance(tensor, torch.Tensor):
            tensor = sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.train_step_context():
            return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            if self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
                # used when calling `trainer.fit`
                return self.model(*args, **kwargs)
            else:
                # used when calling `trainer.validate`
                return self.model.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self.model.test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self.model.predict_step(*args, **kwargs)

    def post_training_step(self):
        if not self.lightning_module.automatic_optimization:
            self.model.require_backward_grad_sync = True

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_spawn_find_unused_parameters_false",
            cls,
            description="DDPSpawn Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def teardown(self) -> None:
        log.detail(f"{self.__class__.__name__}: tearing down strategy")
        super().teardown()

        if isinstance(self.model, DistributedDataParallel):
            if (
                _TORCH_GREATER_EQUAL_1_11
                and not self.model.static_graph
                and self.model._get_ddp_logging_data().get("can_set_static_graph")
            ):
                rank_zero_info(
                    "Your model can run with static graph optimizations. For future training runs, we suggest you"
                    f" pass `Trainer(..., strategy={self.__class__.__name__}(static_graph=True))` to enable them."
                )
            # unwrap model
            self.model = self.lightning_module

        if (
            self.lightning_module.trainer is not None
            and self.lightning_module.trainer.state.fn == TrainerFn.FITTING
            and self._layer_sync
        ):
            # `self.lightning_module.trainer` can be None if teardown gets called on an exception before
            # the trainer gets set on the LightningModule
            self.model = self._layer_sync.revert(self.model)

        if self.root_device.type == "cuda":
            # GPU teardown
            log.detail(f"{self.__class__.__name__}: moving model to CPU")
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
