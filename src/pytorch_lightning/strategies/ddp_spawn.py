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
import logging
import os
from datetime import timedelta
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.distributed
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from typing_extensions import Literal

import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.utilities.distributed import (
    _distributed_available,
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_11
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.types import ReduceOp
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.base import _LightningPrecisionModuleWrapperBase
from pytorch_lightning.overrides.distributed import prepare_for_backward
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.launchers.multiprocessing import _MultiProcessingLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import register_ddp_comm_hook
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.types import PredictStep, STEP_OUTPUT, TestStep, ValidationStep

log = logging.getLogger(__name__)

_DDP_FORK_ALIASES = (
    "ddp_fork",
    "ddp_fork_find_unused_parameters_false",
    "ddp_notebook",
    "ddp_notebook_find_unused_parameters_false",
)


class DDPSpawnStrategy(ParallelStrategy):
    """Spawns processes using the :func:`torch.multiprocessing.spawn` method and joins processes after training
    finishes."""

    strategy_name = "ddp_spawn"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        start_method: Literal["spawn", "fork", "forkserver"] = "spawn",
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
        self._timeout: Optional[timedelta] = timeout
        self._start_method = start_method

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
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def _is_single_process_single_device(self) -> bool:
        return True

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    def _configure_launcher(self) -> None:
        self._launcher = _MultiProcessingLauncher(self, start_method=self._start_method)

    def setup_environment(self) -> None:
        self.setup_distributed()
        super().setup_environment()

    def setup(self, trainer: "pl.Trainer") -> None:

        assert self.cluster_environment is not None
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)

        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        # move the model to the correct device
        self.model_to_device()

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            if self._layer_sync:
                assert self.model is not None
                self.model = self._layer_sync.apply(self.model)

        self.setup_precision_plugin()

        if trainer_fn == TrainerFn.FITTING:
            self.configure_ddp()

    def _setup_model(self, model: Module) -> DistributedDataParallel:
        """Wraps the model into a :class:`~torch.nn.parallel.distributed.DistributedDataParallel` module."""
        return DistributedDataParallel(module=model, device_ids=self.determine_ddp_device_ids(), **self._ddp_kwargs)

    def setup_distributed(self) -> None:
        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        self.set_world_ranks()
        rank_zero_only.rank = self.global_rank
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(
            self.cluster_environment,
            self._process_group_backend,
            self.global_rank,
            self.world_size,
            timeout=self._timeout,
        )

    def set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def pre_configure_ddp(self) -> None:
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)

    def _register_ddp_hooks(self) -> None:
        # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
        # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
        if self.root_device.type == "cuda" and self._is_single_process_single_device:
            assert isinstance(self.model, DistributedDataParallel)
            register_ddp_comm_hook(
                model=self.model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )

    def configure_ddp(self) -> None:
        self.pre_configure_ddp()
        assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
        self.model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()

        # set up optimizers after the wrapped module has been moved to the device
        assert self.lightning_module is not None
        self.setup_optimizers(self.lightning_module.trainer)
        _optimizers_to_device(self.optimizers, self.root_device)

    def determine_ddp_device_ids(self) -> Optional[List[int]]:
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def barrier(self, *args: Any, **kwargs: Any) -> None:
        if not _distributed_available():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_available():
            return obj
        obj = [obj]
        if self.global_rank != src:
            obj = [None]  # type: ignore[list-item]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def model_to_device(self) -> None:
        if self.root_device.type == "cuda":
            # set the device on the spawned subprocesses
            torch.cuda.set_device(self.root_device)
        assert self.model is not None
        self.model.to(self.root_device)

    def pre_backward(self, closure_loss: Tensor) -> None:
        """Run before precision plugin executes backward."""
        if not isinstance(self.model, DistributedDataParallel):
            return
        assert self.lightning_module is not None
        if not self.lightning_module.automatic_optimization:
            prepare_for_backward(self.model, closure_loss)

    def reduce(
        self, tensor: Tensor, group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean"
    ) -> Tensor:
        """Reduces a tensor from several distributed processes to one aggregated tensor.

        Args:
            tensor: the tensor to sync and reduce
            group: the process group to gather results from. Defaults to all processes (world)
            reduce_op: the reduction operation. Defaults to 'mean'/'avg'.
                Can also be a string 'sum' to calculate the sum during reduction.

        Return:
            reduced value, except when the input was not a tensor the output remains is unchanged
        """
        if isinstance(tensor, Tensor):
            tensor = _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.model is not None
        with self.precision_plugin.train_step_context():
            return self.model(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            assert self.lightning_module is not None
            assert self.model is not None
            if self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
                # used when calling `trainer.fit`
                return self.model(*args, **kwargs)
            else:
                # used when calling `trainer.validate`
                assert isinstance(self.model, ValidationStep)
                return self.model.validation_step(*args, **kwargs)

    def test_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            assert isinstance(self.model, TestStep)
            return self.model.test_step(*args, **kwargs)

    def predict_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            assert isinstance(self.model, PredictStep)
            return self.model.predict_step(*args, **kwargs)

    def post_training_step(self) -> None:
        assert self.lightning_module is not None
        if not self.lightning_module.automatic_optimization:
            assert self.model is not None
            self.model.require_backward_grad_sync = True  # type: ignore[assignment]

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        entries = (
            ("ddp_spawn", "spawn"),
            ("ddp_fork", "fork"),
            ("ddp_notebook", "fork"),
        )
        for name, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `start_method` '{start_method}'",
                start_method=start_method,
            )

        entries = (
            ("ddp_spawn_find_unused_parameters_false", "spawn"),
            ("ddp_fork_find_unused_parameters_false", "fork"),
            ("ddp_notebook_find_unused_parameters_false", "fork"),
        )
        for name, start_method in entries:
            strategy_registry.register(
                name,
                cls,
                description=f"DDP strategy with `find_unused_parameters` as False and `start_method` '{start_method}'",
                find_unused_parameters=False,
                start_method=start_method,
            )

    def teardown(self) -> None:
        log.detail(f"{self.__class__.__name__}: tearing down strategy")

        pl_module = self.lightning_module
        if isinstance(self.model, DistributedDataParallel):
            if (
                _TORCH_GREATER_EQUAL_1_11
                and not self.model.static_graph
                and self.model._get_ddp_logging_data().get("can_set_static_graph")  # type: ignore[operator]
            ):
                rank_zero_info(
                    "Your model can run with static graph optimizations. For future training runs, we suggest you"
                    f" pass `Trainer(..., strategy={self.__class__.__name__}(static_graph=True))` to enable them."
                )
            # unwrap model
            self.model = pl_module

        if (
            pl_module is not None
            # `self.lightning_module._trainer` can be None if teardown gets called on an exception before
            # the trainer gets set on the LightningModule
            and pl_module._trainer is not None
            and pl_module._trainer.state.fn == TrainerFn.FITTING
            and self._layer_sync
        ):
            assert self.model is not None
            self.model = self._layer_sync.revert(self.model)
        super().teardown()
