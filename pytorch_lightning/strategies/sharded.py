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
import shutil
import signal
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import torch
import torch.distributed
from torch.nn import Module
from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.distributed import group as _group
from pytorch_lightning.utilities.distributed import init_dist_connection, ReduceOp, sync_ddp_if_available
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import DeadlockDetectedException, MisconfigurationException
from pytorch_lightning.utilities.imports import (
    _FAIRSCALE_AVAILABLE,
    _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE,
    _TORCH_GREATER_EQUAL_1_8,
)
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import STEP_OUTPUT

if _FAIRSCALE_AVAILABLE:
    from fairscale.nn.data_parallel.sharded_ddp import ShardedDataParallel
    from fairscale.optim import OSS

    from pytorch_lightning.overrides.fairscale import LightningShardedDataParallel, unwrap_lightning_module_sharded

log = logging.getLogger(__name__)


class DDPShardedStrategy(ParallelStrategy):
    """Optimizer and gradient sharded training provided by FairScale."""

    strategy_name = "ddp_sharded"
    _REDUCE_BUFFER_SIZE_DEFAULT: int = 2**23  # 8M

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        **kwargs: Union[Any, Dict[str, Any]],
    ):
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self._num_nodes: int = 1
        self._rank_0_will_call_children_scripts: bool = False
        self._kwargs = kwargs
        self.set_world_ranks()

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @property
    def root_device(self) -> torch.device:
        return self.parallel_devices[self.local_rank]

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def distributed_sampler_kwargs(self) -> Dict[str, int]:
        distributed_sampler_kwargs = {"num_replicas": len(self.parallel_devices), "rank": self.global_rank}
        return distributed_sampler_kwargs

    def setup_environment(self) -> None:
        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        # determine which process we are and world size
        self.set_world_ranks()

        init_dist_connection(self.cluster_environment, self.torch_distributed_backend)
        super().setup_environment()

    def setup(self, trainer: "pl.Trainer") -> None:
        super().setup(trainer)
        # share pids to all processes
        self._rank_0_will_call_children_scripts = self.broadcast(self._rank_0_will_call_children_scripts)
        if self._should_run_deadlock_detection():
            self._share_information_to_prevent_deadlock()

        # move the model to the correct device
        self.model_to_device()

        if self._layer_sync:
            self.model = self._layer_sync.apply(self.model)

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            self._configure_sdp(trainer)

    def model_to_device(self) -> None:
        log.detail(f"{self.__class__.__name__}: moving model to device [{self.root_device}]...")
        self.model.to(self.root_device)

    def set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

    def _configure_sdp(self, trainer: "pl.Trainer") -> None:
        if "reduce_buffer_size" not in self._kwargs:
            # For multi-node training, enabling bucketing will improve performance.
            self._kwargs["reduce_buffer_size"] = self._REDUCE_BUFFER_SIZE_DEFAULT if self.num_nodes > 1 else 0

        self.model, self.optimizers = self._setup_model_and_optimizers(
            model=LightningShardedDataParallel(self.model),
            optimizers=trainer.optimizers,
        )

    def barrier(self, *args, **kwargs) -> None:
        if not distributed_available():
            return
        if _TORCH_GREATER_EQUAL_1_8 and torch.distributed.get_backend() == "nccl":
            if self.root_device.type == "cpu":
                device_ids = None
            else:
                device_ids = [self.root_device.index]

            torch.distributed.barrier(device_ids=device_ids)
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        obj = [obj]
        if self.global_rank != src:
            obj = [None]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

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
            if isinstance(self.model, ShardedDataParallel):
                # used when calling `trainer.fit`
                return self.model(*args, **kwargs)
            else:
                # used when calling `trainer.validate`
                return self.lightning_module.validation_step(*args, **kwargs)

    def _setup_model_and_optimizers(self, model: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
        """Wraps the model and optimizers with fairscale components.

        Return:
            The model wrapped into a :class:`~fairscale.nn.data_parallel.ShardedDataParallel` module
            and a list of optimizer wrapped in :class:~`fairscale.optim.OSS`.
        """
        optimizers = self._wrap_optimizers(optimizers)
        model = ShardedDataParallel(model, sharded_optimizer=optimizers, **self._kwargs)
        return model, optimizers

    def _reinit_optimizers_with_oss(self, optimizers: List[Union[Optimizer, LightningOptimizer]]) -> List["OSS"]:
        for x, optimizer in enumerate(optimizers):
            if isinstance(optimizer, LightningOptimizer):
                optimizer = optimizer._optimizer
            if not isinstance(optimizer, OSS):
                optim_class = type(optimizer)
                zero_optimizer = OSS(params=optimizer.param_groups, optim=optim_class, **optimizer.defaults)
                if _FAIRSCALE_OSS_FP16_BROADCAST_AVAILABLE:
                    is_fp16 = self.precision_plugin.precision in (PrecisionType.MIXED, PrecisionType.HALF)
                    # For multi-node training, compressing the model shards in fp16 before broadcasting
                    # improves performance. When using PyTorch AMP, it will not degrade
                    # the model performance.
                    zero_optimizer.broadcast_fp16 = is_fp16 and self.num_nodes > 1
                optimizers[x] = zero_optimizer
                del optimizer
        return optimizers

    def _wrap_optimizers(self, optimizers: List[Optimizer]) -> List["OSS"]:
        if self.model is not None and self.model.trainer.state.fn != TrainerFn.FITTING:
            return optimizers

        return self._reinit_optimizers_with_oss(optimizers)

    def optimizer_state(self, optimizer: "OSS") -> Optional[dict]:
        if isinstance(optimizer, LightningOptimizer):
            optimizer = optimizer._optimizer
        optimizer.consolidate_state_dict()
        return self._optim_state_dict(optimizer)

    @rank_zero_only
    def _optim_state_dict(self, optimizer):
        """
        Retrieves state dict only on rank 0, which contains the entire optimizer state after calling
        :meth:`consolidate_state_dict`.
        """
        return optimizer.state_dict()

    @property
    def lightning_module(self) -> Optional["pl.LightningModule"]:
        if not _FAIRSCALE_AVAILABLE:  # pragma: no cover
            raise MisconfigurationException(
                "`DDPShardedStrategy` requires `fairscale` to be installed."
                " Install it by running `pip install fairscale`."
            )
        return unwrap_lightning_module_sharded(self.model) if self.model is not None else None

    @contextmanager
    def block_backward_sync(self) -> Generator:
        """Blocks syncing gradients behaviour on backwards pass.

        This is useful for skipping sync when accumulating gradients, reducing communication overhead
        Returns: context manager with sync behaviour off
        """
        if isinstance(self.model, ShardedDataParallel):
            with self.model.no_sync():
                yield None
        else:
            yield None

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "ddp_sharded_find_unused_parameters_false",
            cls,
            description="DDP Sharded Strategy with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )
        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )

    def _should_run_deadlock_detection(self) -> bool:
        """Determines whether the plugin will perform process reconciliation in case of errors.

        If the environment variable `PL_RECONCILE_PROCESS` is set, run detection regardless of the cluster environment.
        By default this is disabled. Otherwise, if the cluster environment creates the processes, allow the scheduler /
        parent process to perform the process termination, external to Lightning.
        """
        return os.getenv("PL_RECONCILE_PROCESS", "0") == "1" or self._rank_0_will_call_children_scripts

    def reconciliate_processes(self, trace: str) -> None:
        if self.world_size < 2:
            return

        if not self._should_run_deadlock_detection():
            return

        sync_dir = self._sync_dir

        if not sync_dir:
            rank_zero_warn("Error handling mechanism for deadlock detection is uninitialized. Skipping check.")
            return

        # The cluster may be configured to periodically purge the `/tmp`
        # directory, in which case `sync_dir` may not exist anymore at this
        # point. Idempotently create it to ensure its existence.
        Path(sync_dir).mkdir(parents=True, exist_ok=True)

        # save a file locally.
        torch.save(True, os.path.join(sync_dir, f"{self.global_rank}.pl"))

        # sleep for a short time
        time.sleep(3)

        # return if all processes wrote a file in the `sync_dir`.
        # todo (tchaton) Add support for non-shared file-system which will fail.
        if len(os.listdir(sync_dir)) == (self.world_size // self.num_nodes):
            return

        for pid in self._pids:
            if pid != os.getpid():
                os.kill(pid, signal.SIGKILL)
        shutil.rmtree(sync_dir)
        raise DeadlockDetectedException(f"DeadLock detected from rank: {self.global_rank} \n {trace}")

    def teardown(self) -> None:
        log.detail(f"{self.__class__.__name__}: tearing down DDP sharded plugin")
        super().teardown()
        if self._layer_sync:
            self.model = self._layer_sync.revert(self.model)

        if self.root_device.type == "cuda":
            # GPU teardown
            log.detail(f"{self.__class__.__name__}: moving model to CPU")
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
