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
import re
from multiprocessing.queues import SimpleQueue
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
import torch.distributed
import torch.multiprocessing as mp
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel

import pytorch_lightning as pl
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.distributed import prepare_for_backward
from pytorch_lightning.overrides.torch_distributed import broadcast_object_list
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.training_type.parallel import ParallelPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import (
    _TORCH_GREATER_EQUAL_1_7,
    _TORCH_GREATER_EQUAL_1_8,
    rank_zero_deprecation,
    rank_zero_warn,
)
from pytorch_lightning.utilities.apply_func import apply_to_collection, move_data_to_device
from pytorch_lightning.utilities.cloud_io import atomic_save
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.utilities.distributed import _revert_sync_batchnorm, distributed_available
from pytorch_lightning.utilities.distributed import group as _group
from pytorch_lightning.utilities.distributed import (
    init_dist_connection,
    rank_zero_only,
    ReduceOp,
    sync_ddp_if_available,
)
from pytorch_lightning.utilities.enums import DistributedType
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import STEP_OUTPUT

if _TORCH_GREATER_EQUAL_1_8:
    from pytorch_lightning.utilities.distributed import register_ddp_comm_hook

log = logging.getLogger(__name__)


class DDPSpawnPlugin(ParallelPlugin):
    """Spawns processes using the :func:`torch.multiprocessing.spawn` method and joins processes after training
    finishes."""

    distributed_backend = DistributedType.DDP_SPAWN

    def __init__(
        self,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: Optional[int] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        sync_batchnorm: Optional[bool] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[callable] = None,
        ddp_comm_wrapper: Optional[callable] = None,
        **kwargs: Any,
    ):
        super().__init__(
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
        )
        if num_nodes is not None:
            rank_zero_deprecation(
                "Argument `num_nodes` in `DDPSpawnPlugin` is deprecated in v1.4, and will be removed in v1.6. "
                "Notice that it will be overriden by the trainer setting."
            )
        self._num_nodes = num_nodes or 1
        if sync_batchnorm is not None:
            rank_zero_deprecation(
                "Argument `sync_batchnorm` in `DDPSpawnPlugin` is deprecated in v1.4, and will be removed in v1.6. "
                "Notice that it will be overriden by the trainer setting."
            )
        self._sync_batchnorm = sync_batchnorm or False
        self._ddp_kwargs = kwargs
        self.num_processes = len(parallel_devices) if parallel_devices is not None else 0
        self.mp_queue = None
        self._ddp_comm_state = ddp_comm_state
        self._ddp_comm_hook = ddp_comm_hook
        self._ddp_comm_wrapper = ddp_comm_wrapper
        self._local_rank = 0
        self.set_world_ranks()

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes
        self.set_world_ranks()

    @property
    def sync_batchnorm(self) -> bool:
        return self._sync_batchnorm

    @sync_batchnorm.setter
    def sync_batchnorm(self, sync_batchnorm: bool) -> None:
        self._sync_batchnorm = sync_batchnorm

    @property
    def local_rank(self) -> int:
        return self._local_rank

    def __getstate__(self):
        """Makes this plugin pickleable without destroying the queue in the current process."""
        state = self.__dict__.copy()
        state["mp_queue"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state

    @property
    def root_device(self):
        return self.parallel_devices[self.local_rank]

    @property
    def distributed_sampler_kwargs(self):
        distributed_sampler_kwargs = dict(num_replicas=(self.num_nodes * self.num_processes), rank=self.global_rank)
        return distributed_sampler_kwargs

    @property
    def _is_single_process_single_device(self):
        return True

    def setup(self) -> None:
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        # pass in a state q
        smp = mp.get_context("spawn")
        self.mp_queue = smp.SimpleQueue()

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

    def get_mp_spawn_kwargs(self, trainer: Optional["pl.Trainer"] = None) -> Dict[str, Any]:
        return {"nprocs": self.num_processes}

    def start_training(self, trainer: "pl.Trainer") -> None:
        self.spawn(self.new_process, trainer, self.mp_queue, return_result=False)
        # reset optimizers, since main process is never used for training and thus does not have a valid optim state
        trainer.optimizers = []

    def start_evaluating(self, trainer: "pl.Trainer") -> None:
        self.spawn(self.new_process, trainer, self.mp_queue, return_result=False)

    def start_predicting(self, trainer: "pl.Trainer") -> None:
        self.spawn(self.new_process, trainer, self.mp_queue, return_result=False)

    def spawn(self, function: Callable, *args: Any, return_result: bool = True, **kwargs: Any) -> Optional[Any]:
        """Spawn processes that run the given function.

        Args:
            function: The function to spawn processes from.
            *args: Optional positional arguments that will be passed to the function in addition to the process index.
                These arguments must be pickleable.
            return_result: If ``True``, copies the output of the function from process 0 to the main process and
                returns it.
            **kwargs: Optional named arguments that will be passed to the function in addition to the process index.
                These arguments must be pickleable.

        Return:
            The output of the function of process 0.
        """
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        context = mp.get_context("spawn")
        return_queue = context.SimpleQueue() if return_result else None
        mp.spawn(self._wrapped_function, args=(function, args, kwargs, return_queue), nprocs=self.num_processes)
        return return_queue.get() if return_result else None

    def _wrapped_function(
        self, process_idx: int, function: Callable, args: Any, kwargs: Any, return_queue: Optional[SimpleQueue]
    ) -> None:
        self._worker_setup(process_idx)
        result = function(*args, **kwargs)
        if return_queue is not None and self.local_rank == 0:
            return_queue.put(move_data_to_device(result, "cpu"))

    def _worker_setup(self, process_idx: int):
        reset_seed()
        self.set_world_ranks(process_idx)
        rank_zero_only.rank = self.global_rank
        init_dist_connection(
            self.cluster_environment, self.torch_distributed_backend, self.global_rank, self.world_size
        )

    def new_process(self, trainer: "pl.Trainer", mp_queue: SimpleQueue) -> None:
        self.mp_queue = mp_queue

        # move the model to the correct device
        self.model_to_device()

        if self.sync_batchnorm:
            self.model = self.configure_sync_batchnorm(self.model)

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = self.lightning_module.trainer.state.fn
        if trainer_fn == TrainerFn.FITTING:
            self.configure_ddp()

        self.barrier()

        results = trainer.run_stage()

        # persist info in ddp_spawn
        self.__transfer_distrib_spawn_state_on_fit_end(trainer, results)

        # ensure that spawned processes go through teardown before joining
        trainer._call_teardown_hook()

    def post_dispatch(self, trainer: "pl.Trainer"):
        # restore main state with best weights
        best_path = self.mp_queue.get()
        last_path = self.mp_queue.get()
        self._results = self.mp_queue.get()
        # get the `callback_metrics` and set it to the trainer
        # only in case the user does not override it.
        # TODO: Remove the if in v1.7
        if is_overridden("get_from_queue", self.lightning_module):
            self.lightning_module.get_from_queue(self.mp_queue)
        else:
            self.get_from_queue(trainer, self.mp_queue)

        # recover the weights of the processes trained in the children
        self.__recover_child_process_weights(best_path, last_path)

    def pre_configure_ddp(self):
        # if unset, default `find_unused_parameters` `True`
        # Many models require setting this parameter to True, as there are corner cases
        # when not all parameter backward hooks are fired by the autograd engine even if require_grad is set to True.
        # This flag does come with a performance hit, so it is suggested to disable in cases where it is possible.
        self._ddp_kwargs["find_unused_parameters"] = self._ddp_kwargs.get("find_unused_parameters", True)
        # todo: PyTorch 1.7.0 DDP introduces `self.reducer._rebuild_buckets()` breaking manual_optimization
        if (
            _TORCH_GREATER_EQUAL_1_7
            and not self.lightning_module.automatic_optimization
            and not self._ddp_kwargs.get("find_unused_parameters", False)
        ):
            rank_zero_warn(
                "From PyTorch 1.7.0, Lightning ``manual_optimization`` needs to set ``find_unused_parameters=True`` "
                "to properly work with DDP."
            )
            self._ddp_kwargs["find_unused_parameters"] = True

    def _register_ddp_hooks(self) -> None:
        # currently, DDP communication hooks only work with NCCL backend and SPSD (single process single device) mode
        # https://github.com/pytorch/pytorch/blob/v1.8.0/torch/nn/parallel/distributed.py#L1080-L1084
        if _TORCH_GREATER_EQUAL_1_8 and self.on_gpu and self._is_single_process_single_device:
            register_ddp_comm_hook(
                model=self._model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )

    def configure_ddp(self) -> None:
        self.pre_configure_ddp()
        self._model = self._setup_model(LightningDistributedModule(self.model))
        self._register_ddp_hooks()

    def determine_ddp_device_ids(self):
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def __transfer_distrib_spawn_state_on_fit_end(self, trainer: "pl.Trainer", results: Any) -> None:
        checkpoint_callback = trainer.checkpoint_callback
        best_model_path = checkpoint_callback.best_model_path if checkpoint_callback else None

        # requires to compute the state_dict on all processes in case Metrics are present
        state_dict = self.lightning_module.state_dict()

        if self.global_rank == 0 and self.mp_queue is not None:
            rank_zero_warn("cleaning up ddp environment...")

            # save the last weights
            last_path = None
            if trainer.state.fn == TrainerFn.FITTING and best_model_path is not None and len(best_model_path) > 0:
                last_path = re.sub(".ckpt", ".tmp_end.ckpt", best_model_path)
                atomic_save(state_dict, last_path)

            # todo, pass complete checkpoint as state dictionary
            self.mp_queue.put(best_model_path)
            self.mp_queue.put(last_path)
            self.mp_queue.put(results)
            # adds the `callback_metrics` to the queue
            # TODO: Remove the if in v1.7
            if is_overridden("add_to_queue", self.lightning_module):
                self.lightning_module.add_to_queue(self.mp_queue)
            else:
                self.add_to_queue(trainer, self.mp_queue)

    def __recover_child_process_weights(self, best_path, last_path):
        # transfer back the best path to the trainer
        if self.lightning_module.trainer.checkpoint_callback:
            self.lightning_module.trainer.checkpoint_callback.best_model_path = best_path
        # todo, pass also best score

        # load last weights
        if last_path is not None and self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
            ckpt = pl_load(last_path, map_location=lambda storage, loc: storage)
            self.lightning_module.load_state_dict(ckpt)
            self.checkpoint_io.remove_checkpoint(last_path)

    def barrier(self, *args, **kwargs) -> None:
        if not distributed_available():
            return
        if _TORCH_GREATER_EQUAL_1_8 and torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: object, src: int = 0) -> object:
        if not distributed_available():
            return obj
        obj = [obj]
        if self.global_rank != src:
            obj = [None]
        broadcast_object_list(obj, src, group=_group.WORLD)
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

    def training_step(self, *args, **kwargs) -> Optional[Any]:
        return self.model(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        if isinstance(self.model, DistributedDataParallel):
            # used when calling `trainer.fit`
            return self.model(*args, **kwargs)
        else:
            # used when calling `trainer.validate`
            return self.lightning_module.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return self.lightning_module.test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> Any:
        return self.lightning_module.predict_step(*args, **kwargs)

    def post_training_step(self):
        if not self.lightning_module.automatic_optimization:
            self.model.require_backward_grad_sync = True

    def add_to_queue(self, trainer: "pl.Trainer", queue: torch.multiprocessing.SimpleQueue) -> None:
        """Appends the :attr:`trainer.callback_metrics` dictionary to the given queue. To avoid issues with memory
        sharing, we cast the data to numpy.

        Args:
            queue: the instance of the queue to append the data.
        """
        callback_metrics: dict = apply_to_collection(
            trainer.callback_metrics, torch.Tensor, lambda x: x.cpu().numpy()
        )  # send as numpy to avoid issues with memory sharing
        queue.put(callback_metrics)

    def get_from_queue(self, trainer: "pl.Trainer", queue: torch.multiprocessing.SimpleQueue) -> None:
        """Retrieve the :attr:`trainer.callback_metrics` dictionary from the given queue. To preserve consistency,
        we cast back the data to ``torch.Tensor``.

        Args:
            queue: the instance of the queue from where to get the data.
        """
        # NOTE: `add_to_queue` needs to be called before
        callback_metrics: dict = queue.get()
        trainer.callback_metrics.update(apply_to_collection(callback_metrics, np.ndarray, lambda x: torch.tensor(x)))

    @classmethod
    def register_plugins(cls, plugin_registry: Dict) -> None:
        plugin_registry.register(
            "ddp_spawn_find_unused_parameters_false",
            cls,
            description="DDPSpawn Plugin with `find_unused_parameters` as False",
            find_unused_parameters=False,
        )

    def teardown(self) -> None:
        if isinstance(self.model, DistributedDataParallel):
            self.model = self.lightning_module

        if self.sync_batchnorm:
            self.model = _revert_sync_batchnorm(self.model)

        if self.on_gpu:
            # GPU teardown
            self.lightning_module.cpu()
            # clean up memory
            torch.cuda.empty_cache()
