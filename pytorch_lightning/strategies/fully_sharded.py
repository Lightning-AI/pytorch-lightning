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
import contextlib
import logging
import os
import shutil
import signal
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

import torch
import torch.distributed

import pytorch_lightning as pl
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.utilities import _FAIRSCALE_FULLY_SHARDED_AVAILABLE, _TORCH_GREATER_EQUAL_1_8
from pytorch_lightning.utilities.distributed import distributed_available
from pytorch_lightning.utilities.distributed import group as _group
from pytorch_lightning.utilities.distributed import init_dist_connection, ReduceOp, sync_ddp_if_available
from pytorch_lightning.utilities.enums import PrecisionType
from pytorch_lightning.utilities.exceptions import DeadlockDetectedException, MisconfigurationException
from pytorch_lightning.utilities.optimizer import optimizers_to_device
from pytorch_lightning.utilities.rank_zero import rank_zero_only, rank_zero_warn
from pytorch_lightning.utilities.seed import reset_seed
from pytorch_lightning.utilities.types import STEP_OUTPUT

if _FAIRSCALE_FULLY_SHARDED_AVAILABLE:
    from fairscale.nn import default_auto_wrap_policy, enable_wrap
    from fairscale.nn.data_parallel import FullyShardedDataParallel

log = logging.getLogger(__name__)


class DDPFullyShardedStrategy(ParallelStrategy):

    strategy_name = "ddp_fully_sharded"

    def __init__(
        self,
        accelerator: Optional["pl.accelerators.accelerator.Accelerator"] = None,
        cpu_offload: bool = False,
        flatten_parameters: bool = True,
        reshard_after_forward: bool = True,
        move_grads_to_cpu: Optional[bool] = None,
        fp32_reduce_scatter: Optional[bool] = None,
        compute_dtype: Optional[torch.dtype] = None,
        bucket_cap_mb: int = 25,
        min_num_params: int = 1e8,
        state_dict_to_cpu: bool = True,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
    ):
        """Plugin for Fully Sharded Data Parallel provided by FairScale.

        Full Sharded Training shards the entire model across all available GPUs, allowing you to scale model
        size, whilst using efficient communication to reduce overhead. In practice, this means we can remain
        at parity with PyTorch DDP, whilst scaling our model sizes dramatically. The technique is similar
        to ZeRO-Stage 3 but has been built for upstreaming to PyTorch.
        `For more information: https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html`.
        .. warning:: ``FullyShardedPlugin`` is in beta and subject to change.

        Defaults have been set and options have been exposed, but may require configuration
        based on your level of memory/speed efficiency. We suggest having a look at this PR for more information.
        `https://github.com/facebookresearch/fairscale/pull/413`

        Many of the helpful doc strings below came from the original FairScale documentation:
        `https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html`

        Arguments:
            cpu_offload: Offload FP32 params to CPU. Only usable in precision=16 mode.
                (Default: False).
            move_grads_to_cpu: Moves gradient shards to CPU after reduction.
                Only disable if using CPU based optimizers
                (Default to ``cpu_offload``).
            flatten_parameters: Flattens parameter into single contiguous tensor for speed efficiency
                (Default: True).
            reshard_after_forward: Reshard parameters after the forward pass, which saves memory but slows
                down training. This is only relevant when resharding individual layers.
                (Default: True).
            fp32_reduce_scatter: Reduce-Scatter gradients in FP32. Only relevant in mixed precision
                (Default: None).
            compute_dtype: dtype for full parameters for computation. Default to torch.float32,
                unless using mixed precision, in which case defaults to torch.float16.
                (Default: None).
            bucket_cap_mb: bucket parameters so that gradient reduction
                can potentially overlap with backward computation.
                bucket_cap_mb controls the bucket size in MegaBytes (MB).
                Buckets are sub-divided based on world_size,
                so the max shard size is roughly bucket_cap_mb / world_size.
                Values <= 0 disable bucketing.
                (Default: 25).
            min_num_params: Number of parameters to wrap when using FairScale ``auto_wrap``.
                (Default: 1e8)
            state_dict_to_cpu: Whether to return parameters (returned by :func:`state_dict`) on CPU device.
                If ``False``, this will default to ``compute_device``.
                (Default: True).
        """

        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
        )
        self._num_nodes = 1
        self.cpu_offload = cpu_offload
        self.move_grads_to_cpu = move_grads_to_cpu
        self.flatten_parameters = flatten_parameters
        self.reshard_after_forward = reshard_after_forward
        self.fp32_reduce_scatter = fp32_reduce_scatter
        self.compute_dtype = compute_dtype
        self.bucket_cap_mb = bucket_cap_mb
        self.min_num_params = min_num_params
        self.state_dict_device = torch.device("cpu") if state_dict_to_cpu else None
        self._process_group = None
        self._rank_0_will_call_children_scripts: bool = False
        self.set_world_ranks()

    @property
    def process_group(self):
        if self._process_group is None:
            self._process_group = torch.distributed.new_group()
        return self._process_group

    @property
    def root_device(self) -> torch.device:
        return self.parallel_devices[self.local_rank]

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes
        self.set_world_ranks()

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    def setup_environment(self) -> None:
        if not self.root_device.type == "cuda":
            raise MisconfigurationException(
                "You selected strategy to be `ddp_fully_sharded`, but GPU is not available."
            )

        log.detail(f"{self.__class__.__name__}: setting up distributed...")
        reset_seed()

        if self._layer_sync:
            self.model = self._layer_sync.apply(self.model)
        # determine which process we are and world size
        self.set_world_ranks()

        # set warning rank
        rank_zero_only.rank = self.global_rank

        init_dist_connection(self.cluster_environment, self.torch_distributed_backend)

        super().setup_environment()

    @contextlib.contextmanager
    def model_sharded_context(self) -> Generator:
        log.detail(f"{self.__class__.__name__}: entered model_sharded_context.")
        precision = self.precision_plugin.precision

        def wrap_policy(*args, **kwargs):
            return default_auto_wrap_policy(*args, **kwargs, min_num_params=self.min_num_params)

        with enable_wrap(
            wrapper_cls=FullyShardedDataParallel,
            auto_wrap_policy=wrap_policy,
            process_group=self.process_group,
            cpu_offload=self.cpu_offload,
            move_grads_to_cpu=self.move_grads_to_cpu,
            flatten_parameters=self.flatten_parameters,
            mixed_precision=(precision == PrecisionType.MIXED),
            reshard_after_forward=self.reshard_after_forward,
            fp32_reduce_scatter=self.fp32_reduce_scatter,
            compute_dtype=self.compute_dtype,
            bucket_cap_mb=self.bucket_cap_mb,
            state_dict_device=self.state_dict_device,
        ):
            yield

        log.detail(f"{self.__class__.__name__}: exiting model_sharded_context.")

    def setup(self, trainer: "pl.Trainer") -> None:
        self.accelerator.setup(trainer)
        self.setup_optimizers(trainer)
        self.setup_precision_plugin()
        optimizers_to_device(self.optimizers, self.root_device)

        if self._layer_sync:
            self.model = self._layer_sync.apply(self.model)

        log.detail(f"{self.__class__.__name__}: configuring FSDP... (cpu_offload: [{self.cpu_offload}])")
        if not self.cpu_offload:
            # When using CPU Offload, FSDP will manage the CUDA movement for us.
            # Note: this would be problematic for large model (which could not fit in one GPU)
            # as FSDP module.to(device) would first summon all parameters
            # (TODO: need to figure out solution)
            self.model_to_device()

        # setup optimizers after fully sharded has wrapped the lightning module
        self.barrier()
        self.setup_optimizers(trainer)

    def model_to_device(self) -> None:
        log.detail(f"{self.__class__.__name__}: moving model to device [{self.root_device}]...")
        # ensure we update the device type in the lightning module
        self.lightning_module.to(self.root_device)

    def _configure_launcher(self) -> None:
        self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)
        if not self.cluster_environment.creates_processes_externally:
            self._rank_0_will_call_children_scripts = True

    def set_world_ranks(self) -> None:
        if self.cluster_environment is None:
            return
        self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
        self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        rank_zero_only.rank = self.cluster_environment.global_rank()

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

    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.train_step_context():
            return self.model.training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.val_step_context():
            return self.model.validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        with self.precision_plugin.test_step_context():
            return self.model.test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs) -> STEP_OUTPUT:
        with self.precision_plugin.predict_step_context():
            return self.model.predict_step(*args, **kwargs)

    @classmethod
    def register_strategies(cls, strategy_registry: Dict) -> None:
        strategy_registry.register(
            "fsdp", cls, description="Fully sharded training with checkpointing the full state dict."
        )

        strategy_registry.register(
            cls.strategy_name,
            cls,
            description=f"{cls.__class__.__name__}",
        )
