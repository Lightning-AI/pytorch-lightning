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
# limitations under the License
import logging
import os
from typing import Callable, List, Optional

import torch
import torch.distributed as torch_distrib
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.overrides.distributed import LightningDistributedModule
from pytorch_lightning.plugins.training_type.rpc import DEFAULT_RPC_TIMEOUT_SEC, RPCPlugin
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import _FAIRSCALE_PIPE_AVAILABLE, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _FAIRSCALE_PIPE_AVAILABLE:
    import fairscale.nn.model_parallel as mpu
    from fairscale.nn import PipeRPCWrapper
    from fairscale.nn.pipe import balance as pipe_balance
    from fairscale.nn.pipe import rpc as rpc_pipe
    from fairscale.nn.pipe.pipeline import PipelineStyle

log = logging.getLogger(__name__)


class RPCSequentialPlugin(RPCPlugin):

    def __init__(
        self,
        balance: Optional[List[int]] = None,
        microbatches: int = 8,
        checkpoint: str = 'except_last',
        balance_mode: str = "balance_by_size",
        pipelined_backward: Optional[bool] = True,
        rpc_timeout_sec: float = DEFAULT_RPC_TIMEOUT_SEC,
        **kwargs
    ):
        """
        Provides sequential model parallelism for :class:`nn.Sequential <torch.nn.Sequential>` module.
        If the module requires lots of memory, Pipe can be used to reduce this by leveraging multiple GPUs.

        .. _RPCSequentialPlugin: https://arxiv.org/abs/1811.06965

        Pipeline parallelism comes with with checkpointing to reduce peak
        memory required to train while minimizing device under-utilization.
        This is turned on by default and can be turned off via the checkpoint argument.

        You should determine the balance when defining the plugin,
        or you can pass an example input array via the LightningModule to infer a balance.
        The module will be partitioned into multiple devices according to the given balance. You may also rely on
        your own heuristics to find your own optimal configuration.

        Args:
            balance: The balance of the model, i.e [2, 2] (two layers on each GPU).
            If not provided assumes user provides an input example array to find a balance on all GPUs.

            microbatches: Allows for parallelization to reduce device utilization
            by splitting the batch into further smaller batches.

            checkpoint: Enables gradient checkpointing. ['always', 'except_last', 'never']

            balance_mode: Type of balance heuristic to use if balance to be inferred.

                - 'balance_by_size': checks memory usage of each layer and determines balance

                - 'balance_by_time': checks time of each layer and determines balance

            pipelined_backward: if True, call torch.autograd.backward once per microbatch on the

            backward pass (instead of once for the whole batch). This works
            around a potential deadlock in pytorch when using tensor parallelism
            at the same time. Defaults to `True` if
            `get_model_parallel_world_size() > 1`
        """
        self._check_pipe_available()
        super().__init__(rpc_timeout_sec=rpc_timeout_sec, **kwargs)

        self.balance = balance

        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self.balance_mode = balance_mode
        self.pipelined_backward = pipelined_backward
        self._main_rpc_process = True

    def init_ddp_connection(
        self,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        if self.lightning_module.trainer.amp_backend is not None:
            raise MisconfigurationException(
                '`RPCSequentialPlugin` is currently not supported in Automatic Mixed Precision'
            )

        if self._skip_init_connections():
            return

        global_rank = global_rank if global_rank is not None else self.cluster_environment.global_rank()
        world_size = world_size if world_size is not None else self.cluster_environment.world_size()
        super().init_ddp_connection(global_rank, world_size)
        super().init_rpc_connection(global_rank=global_rank, world_size=world_size)
        model = self.lightning_module
        self.gpus_per_model = self._infer_check_num_gpus()
        self.init_model_parallel_groups()
        self.set_main_rpc_process()

        self._check_sequential_model_exists(model)

        # check if user given balance is valid
        if self.balance is not None:
            self._assert_valid_model_balance()

        if self.main_rpc_process:
            if self.balance is None:
                self._infer_model_balance()
            self.init_pipe_module()
        else:
            self.handle_transferred_pipe_module()
            self.exit_rpc_process()

    def _infer_model_balance(self):
        log.info(f'Inferring model balance using {self.balance_mode} mode')
        model = self.lightning_module
        if model.example_input_array is None:
            raise MisconfigurationException(
                'Please set example_input_array to your model, so we can infer the right model balance for you'
            )
        balance_func = getattr(pipe_balance, self.balance_mode)
        self.balance = balance_func(self.gpus_per_model, model.sequential_module, model.example_input_array)
        self._sync_balance_to_all_parallel_groups()

        log.info(f'The following model balance {self.balance.tolist()} was inferred using {self.balance_mode} mode')

    def _sync_balance_to_all_parallel_groups(self, main_rank=0):
        """
        Ensures that we sync the balance to all main processes, so that the balance is the same per replica.
        Args:
            main_rank: The rank with the balance we'd like to replicate.
        """
        self.balance = torch.tensor(self.balance, dtype=torch.int, device='cuda')
        # Ensure we sync to all processes within the main data parallel group
        # We use the data parallel group as all main processes are found within the same group
        torch_distrib.broadcast(self.balance, src=main_rank, group=mpu.get_data_parallel_group())
        self.balance = self.balance.cpu()

    def _check_sequential_model_exists(self, model):
        if not hasattr(model, "sequential_module") or not isinstance(model.sequential_module, nn.Sequential):
            raise MisconfigurationException(
                'Could not find a PipeLightningModule within the model. '
                'Did you set your sequential model as the `sequential_module` attribute of your model?'
            )

    def _find_and_init_pipe_module(self, model):
        if hasattr(model, "sequential_module") and isinstance(model.sequential_module, LightningPipeModule):
            # model has been wrapped already
            return
        elif hasattr(model, "sequential_module") and isinstance(model.sequential_module, nn.Sequential):
            # try to wrap model for the user
            model.sequential_module = LightningPipeModule(
                model.sequential_module,
                balance=self.balance,
                microbatches=self.microbatches,
                checkpoint=self.checkpoint,
            )
            # Update references for workers to access correct lightning functions when calling RPC
            model.sequential_module.trainer = model.trainer
            model.sequential_module.configure_optimizers = model.configure_optimizers

            # Update references for main process to access correct lightning functions when calling RPC
            model.sequential_module.module.model.trainer = model.trainer
            model.sequential_module.module.model.configure_optimizers = model.configure_optimizers

            self.model = model

        else:
            raise MisconfigurationException(
                'Could not find a PipeLightningModule within the model. '
                'Did you defined set your sequential model as a `sequential_module` attribute of your model?'
            )

    def _assert_valid_model_balance(self):
        model = self.lightning_module
        if sum(self.balance) != len(model.sequential_module):
            raise MisconfigurationException(
                f'The provided balance sum: {sum(self.balance)} does not'
                f' match your Sequential length: {len(model.sequential_module)}'
            )

    def _skip_init_connections(self):
        """
        Skip initialization if torch is already initialized and we're in testing.
        Returns: Whether to skip initialization

        """
        return torch_distrib.is_initialized() and self.lightning_module.trainer.state.fn != TrainerFn.FITTING

    def init_model_parallel_groups(self):
        num_model_parallel = 1  # TODO currently no support for vertical model parallel
        mpu.initialize_model_parallel(model_parallel_size_=num_model_parallel, pipeline_length=self.gpus_per_model)

    def _infer_check_num_gpus(self):
        """
        Infer the number of GPUs per model.

        Returns: The appropriate balance for the model
        """
        if isinstance(self.balance, list):
            if len(self.balance) != (self.world_size / self.num_nodes):
                raise MisconfigurationException(
                    "Pipe currently only supports splitting the module onto all available GPUs"
                )
            # User has defined a balance for his model
            return len(self.balance)
        # Assume that the user wants to balance his model on all GPUs
        return self.world_size

    def handle_transferred_pipe_module(self) -> None:
        if self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
            torch_distrib.barrier()  # Ensure we await main process initialization
            # Add trainer/configure_optimizers to the pipe model for access in all worker processes
            rpc_pipe.PipeModel.trainer = self.lightning_module.trainer
            del rpc_pipe.PipeModel.trainer.model.sequential_module
            rpc_pipe.PipeModel.trainer.model.sequential_module = rpc_pipe.PipeModel
            rpc_pipe.PipeModel.configure_optimizers = self.lightning_module.configure_optimizers

    def init_pipe_module(self) -> None:
        # Create pipe_module
        model = self.lightning_module
        self._find_and_init_pipe_module(model)
        if self.lightning_module.trainer.state.fn == TrainerFn.FITTING:
            torch_distrib.barrier()  # Ensure we join main process initialization
            model.sequential_module.foreach_worker(register_optimizers, include_self=True)

            # TODO: Move this to the connector

    def pre_backward(self, closure_loss: torch.Tensor, should_accumulate: bool, optimizer: Optimizer, opt_idx: int):
        """Run before precision plugin executes backward"""

    def configure_ddp(self):
        if self.main_rpc_process:
            self.pre_configure_ddp()

            self._model = DistributedDataParallel(
                LightningDistributedModule(self.model),
                device_ids=self.determine_ddp_device_ids(),
                process_group=mpu.get_data_parallel_group(),
                **self._ddp_kwargs,
            )
            # Plugin handle backwards across processes. Currently not supported for DDP + pipe parallel
            self._model.require_backward_grad_sync = False

    @rank_zero_only
    def rpc_save_model(self, trainer, save_model_fn: Callable, filepath: str) -> None:
        model = self.lightning_module
        if not hasattr(model.sequential_module, "foreach_worker"):
            return
        current_layers = model.sequential_module
        model.sequential_module.foreach_worker(
            save_layers_on_all_rank_zero_workers, {"gpus_per_model": self.gpus_per_model}, include_self=True
        )
        model.sequential_module = load_sequential_from_saved_layers(self.gpus_per_model)
        save_model_fn(trainer, filepath)
        model.sequential_module = current_layers

    def worker_optimizer_step(self, model: LightningModule, opt_idx: int, *args, **kwargs) -> None:
        model.sequential_module.foreach_worker(
            run_optimizer, {
                "opt_idx": opt_idx,
                "args": args,
                "kwargs": kwargs
            }, include_self=False
        )

    @property
    def distributed_sampler_kwargs(self):
        return dict(
            num_replicas=mpu.get_data_parallel_world_size(),
            rank=mpu.get_data_parallel_rank(),
        )

    @property
    def data_parallel_group(self):
        return mpu.get_data_parallel_group()

    def set_main_rpc_process(self):
        self.main_rpc_process = torch_distrib.get_rank(group=mpu.get_pipeline_parallel_group()) == 0

    @property
    def main_rpc_process(self) -> bool:
        return self._main_rpc_process

    @main_rpc_process.setter
    def main_rpc_process(self, is_main_process):
        self._main_rpc_process = is_main_process

    def barrier(self, name: Optional[str] = None) -> None:
        if torch_distrib.is_initialized() and self.main_rpc_process:
            torch_distrib.barrier(group=self.data_parallel_group)

    def _check_pipe_available(self):
        if not _FAIRSCALE_PIPE_AVAILABLE:
            raise MisconfigurationException(
                'PipeRPCPlugin requires FairScale and currently is only supported on PyTorch 1.6.'
            )

    def post_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int, **kwargs) -> None:
        """Hook to do something after each optimizer step."""
        if self.rpc_enabled and self.main_rpc_process:
            # Initialize optimizer step on main process
            self.worker_optimizer_step(model=self.lightning_module, opt_idx=optimizer_idx, **kwargs)

    def post_training_step(self):
        if self.main_rpc_process:
            super().post_training_step()

    def start_training(self, trainer) -> None:
        if self.main_rpc_process:
            super().start_training(trainer)

    def start_evaluating(self, trainer) -> None:
        if self.main_rpc_process:
            super().start_evaluating(trainer)


class LightningPipeModule(nn.Module):
    """
    This class wraps Fairscale Pipe and PipeRCPWrapper class.
    """

    def __init__(self, module: nn.Sequential, balance: List[int], microbatches: int = 8, checkpoint='never'):
        super().__init__()
        self.module = module
        self.balance = balance
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self._init_pipe()

    def _init_pipe(self):
        device = torch.device("cuda", torch_distrib.get_rank())

        self.module = PipeRPCWrapper(
            module=self.module,
            balance=self.balance,
            chunks=self.microbatches,
            style=PipelineStyle.MultiProcess,
            input_device=device,
            worker_map=self.get_worker_map(),
            checkpoint=self.checkpoint,
        )

    def foreach_worker(self, *args, **kwargs):
        self.module.foreach_worker(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def get_worker_map(self):
        # TODO, is this correct with multinodes? We also assume "worker" is the same as defined in the RPCPlugin
        return {rank: f"worker{rank}" for rank in range(torch_distrib.get_world_size())}


def register_optimizers(ctx, model):
    optimizers, lr_schedulers, optimizer_frequencies = model.trainer.init_optimizers(model)
    model.trainer.optimizers = optimizers
    model.trainer.lr_schedulers = lr_schedulers
    model.trainer.optimizer_frequencies = optimizer_frequencies


def run_optimizer(ctx, model):
    trainer = model.trainer
    opt_idx = ctx["opt_idx"]
    optimizer = trainer.optimizers[opt_idx]
    optimizer.step(*ctx["args"], **ctx["kwargs"])


def save_layers_on_all_rank_zero_workers(ctx, model):
    gpus_per_model = ctx["gpus_per_model"]
    rank = torch_distrib.get_rank()
    if rank in range(gpus_per_model):
        seq = list(model.children())[0]
        torch.save(seq, f"seq_{rank}.pt")


def load_sequential_from_saved_layers(gpus_per_model):
    partial_seqs = [torch.load(f"seq_{rank}.pt", map_location='cpu') for rank in range(gpus_per_model)]
    seq = nn.Sequential()
    for p_seq in partial_seqs:
        for name, child in p_seq.named_children():
            seq.add_module(name, child)
    # delete tmp files
    [os.remove(f"seq_{rank}.pt") for rank in range(gpus_per_model)]
    return seq
