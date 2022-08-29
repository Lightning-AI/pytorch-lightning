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
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Iterable, Tuple

import torch
import torch.distributed
from torch import Tensor
from torch.distributed.constants import default_pg_timeout
from torch.nn import Module
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.overrides import LightningDistributedModule
from pytorch_lightning.overrides.base import _LightningPrecisionModuleWrapperBase
from pytorch_lightning.overrides.distributed import prepare_for_backward
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.io.checkpoint_plugin import CheckpointIO
from pytorch_lightning.plugins.precision import PrecisionPlugin
from pytorch_lightning.strategies import ParallelStrategy
from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import (
    register_ddp_comm_hook,
)
from pytorch_lightning.utilities.exceptions import DeadlockDetectedException
from pytorch_lightning.utilities.imports import (
    _FAIRSCALE_AVAILABLE,
    _IS_WINDOWS,
    _TORCH_GREATER_EQUAL_1_10,
    _TORCH_GREATER_EQUAL_1_11,
)
from pytorch_lightning.utilities.optimizer import optimizers_to_device
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import PredictStep, STEP_OUTPUT, TestStep, ValidationStep

if _FAIRSCALE_AVAILABLE:
    from fairscale.optim import OSS
else:
    OSS = object
if _TORCH_GREATER_EQUAL_1_10 and torch.distributed.is_available():
    from torch.distributed.algorithms.model_averaging.averagers import ModelAverager


log = logging.getLogger(__name__)

from lightning_lite.lite.accelerators import Accelerator


class DDPStrategy(ParallelStrategy):
    """Strategy for multi-process single-device training on one or multiple nodes."""

    strategy_name = "ddp"

    def __init__(
        self,
        accelerator: Optional["Accelerator"] = None,
        parallel_devices: Optional[List[torch.device]] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        checkpoint_io: Optional[CheckpointIO] = None,
        precision_plugin: Optional[PrecisionPlugin] = None,
        ddp_comm_state: Optional[object] = None,
        ddp_comm_hook: Optional[Callable] = None,
        ddp_comm_wrapper: Optional[Callable] = None,
        model_averaging_period: Optional[int] = None,
        process_group_backend: Optional[str] = None,
        timeout: Optional[timedelta] = default_pg_timeout,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            accelerator=accelerator,
            parallel_devices=parallel_devices,
            cluster_environment=cluster_environment,
            checkpoint_io=checkpoint_io,
            precision_plugin=precision_plugin,
            process_group_backend=process_group_backend,
            timeout=timeout,
            **kwargs,
        )
        self._ddp_comm_state = ddp_comm_state
        self._ddp_comm_hook = ddp_comm_hook
        self._ddp_comm_wrapper = ddp_comm_wrapper
        self._model_averaging_period = model_averaging_period
        self._model_averager: Optional[ModelAverager] = None
        self._pids: List[int] = []
        self._sync_dir: Optional[str] = None
        self._rank_0_will_call_children_scripts: bool = False

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)
            self._rank_0_will_call_children_scripts = True

    def setup(self, trainer: "pl.Trainer") -> None:
        # share ddp pids to all processes
        self._rank_0_will_call_children_scripts = bool(self.broadcast(self._rank_0_will_call_children_scripts))
        if self._should_run_deadlock_detection():
            self._share_information_to_prevent_deadlock()

        assert self.accelerator is not None
        self.accelerator.setup(trainer)

        # move the model to the correct device
        self.module_to_device(self.model)

        # skip wrapping the model if we are not fitting as no gradients need to be exchanged
        trainer_fn = trainer.state.fn

        if trainer_fn == TrainerFn.FITTING:
            if self._layer_sync:
                assert self.model is not None
                self.model = self._layer_sync.apply(self.model)

        self.setup_precision_plugin()

        if trainer_fn == TrainerFn.FITTING:
            self.configure_ddp()

            # set up optimizers after the wrapped module has been moved to the device
            self.setup_optimizers(trainer)
            optimizers_to_device(self.optimizers, self.root_device)

        if _TORCH_GREATER_EQUAL_1_10 and trainer_fn == TrainerFn.FITTING:
            import torch.distributed.algorithms.ddp_comm_hooks.post_localSGD_hook as post_localSGD

            if isinstance(self._ddp_comm_state, post_localSGD.PostLocalSGDState):
                self._enable_model_averaging()

    def _register_ddp_hooks(self) -> None:
        log.detail(f"{self.__class__.__name__}: registering ddp hooks")
        if self.root_device.type == "cuda" and self._is_single_process_single_device:
            assert isinstance(self.model, DistributedDataParallel)
            register_ddp_comm_hook(
                model=self.model,
                ddp_comm_state=self._ddp_comm_state,
                ddp_comm_hook=self._ddp_comm_hook,
                ddp_comm_wrapper=self._ddp_comm_wrapper,
            )

    def _enable_model_averaging(self) -> None:
        # Only called when PyTorch version >= 1.10
        log.detail(f"{self.__class__.__name__}: reinitializing optimizers with post localSGD")
        if self._model_averaging_period is None:
            raise ValueError(
                "Post-localSGD algorithm is used, but model averaging period is not provided to DDP strategy."
            )
        from torch.distributed.optim import DistributedOptimizer, PostLocalSGDOptimizer, ZeroRedundancyOptimizer

        for optimizer in self.optimizers:
            if isinstance(optimizer, LightningOptimizer):
                optimizer = optimizer._optimizer

            is_distributed_optimizer = isinstance(optimizer, DistributedOptimizer) if not _IS_WINDOWS else False
            if (
                is_distributed_optimizer
                or isinstance(optimizer, ZeroRedundancyOptimizer)
                or (_FAIRSCALE_AVAILABLE and isinstance(optimizer, OSS))
                or isinstance(optimizer, PostLocalSGDOptimizer)
            ):
                raise ValueError(
                    f"Currently model averaging cannot work with a distributed optimizer of type "
                    f"{optimizer.__class__.__name__}."
                )

        assert self._ddp_comm_state is not None
        self._model_averager = torch.distributed.algorithms.model_averaging.averagers.PeriodicModelAverager(
            period=self._model_averaging_period, warmup_steps=self._ddp_comm_state.start_localSGD_iter
        )

    def optimizer_step(
        self,
        optimizer: Optimizer,
        opt_idx: int,
        closure: Callable[[], Any],
        model: Optional["pl.LightningModule"] = None,
        **kwargs: Any,
    ) -> Any:
        """Performs the actual optimizer step.

        Args:
            optimizer: the optimizer performing the step
            opt_idx: index of the current optimizer
            closure: closure calculating the loss value
            model: reference to the model, optionally defining optimizer step related hooks
            **kwargs: Any extra arguments to ``optimizer.step``
        """
        optimizer_output = super().optimizer_step(optimizer, opt_idx, closure, model, **kwargs)

        if not _TORCH_GREATER_EQUAL_1_10 or self._model_averager is None:
            return optimizer_output

        params = [param for group in optimizer.param_groups for param in group["params"] if param.grad is not None]
        self._model_averager.average_parameters(iter(params))

        return optimizer_output

    def configure_ddp(self) -> None:
        log.detail(f"{self.__class__.__name__}: configuring DistributedDataParallel")
        assert isinstance(self.model, (pl.LightningModule, _LightningPrecisionModuleWrapperBase))
        self.model = self.setup_module(LightningDistributedModule(self.model))
        self._register_ddp_hooks()

    def pre_backward(self, tensor: Tensor, module: Optional[Module]) -> None:
        if not isinstance(self.model, DistributedDataParallel):
            return
        assert self.lightning_module is not None
        if not self.lightning_module.automatic_optimization:
            prepare_for_backward(self.model, tensor)

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
        strategy_registry.register(
            "ddp_find_unused_parameters_false",
            cls,
            description="DDP Strategy with `find_unused_parameters` as False",
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

    def _share_information_to_prevent_deadlock(self) -> None:
        self._share_pids()

        # there should be a unique sync_dir per nodes.
        if self.local_rank == 0:
            # create a temporary directory used to synchronize processes on deadlock.
            self._sync_dir = tempfile.mkdtemp()

        sync_dirs = []
        global_node_rank_zero = 0
        for _ in range(self.num_nodes):
            sync_dirs.append(self.broadcast(self._sync_dir, global_node_rank_zero))
            global_node_rank_zero += self.world_size // self.num_nodes

        self._sync_dir = sync_dirs[self.node_rank]

    def _share_pids(self) -> None:
        """Make all DDP processes aware of all processes pids."""
        self.barrier()
        pids = self.all_gather(torch.tensor(os.getpid(), device=self.root_device))
        pids = pids.cpu().numpy().tolist()
        self._pids = pids if isinstance(pids, list) else [pids]

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

    def teardown(
            self, modules: Iterable[Module] = (), optimizers: Iterable[Optimizer] = ()
    ) -> Tuple[Iterable[Module], Iterable[Optimizer]]:
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

        return super().teardown()
