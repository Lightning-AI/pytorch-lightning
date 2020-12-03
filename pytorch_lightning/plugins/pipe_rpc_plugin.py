import os
import sys
import weakref
from distutils.version import LooseVersion
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as torch_distrib
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import LightningModule
from pytorch_lightning import _logger as log
from pytorch_lightning.plugins.rpc_plugin import RPCPlugin
from pytorch_lightning.utilities import FAIRSCALE_AVAILABLE, AMPType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

FAIRSCALE_AVAILABLE &= LooseVersion(torch.__version__) == LooseVersion("1.6.0")

if FAIRSCALE_AVAILABLE:
    import fairscale.nn.model_parallel as mpu
    from fairscale.nn import LazyModule, PipeRPCWrapper
    from fairscale.nn.model_parallel.utils import ensure_divisibility
    from fairscale.nn.pipe import balance as pipe_balance
    from fairscale.nn.pipe import rpc as rpc_pipe
    from fairscale.nn.pipe.pipeline import PipelineStyle
    from torch.distributed import rpc


def get_worker_map():
    # TODO, is this correct with multinodes?
    return {rank: f"worker{rank}" for rank in range(torch_distrib.get_world_size())}


def register_optimizers(ctx, model):
    optimizers, lr_schedulers, optimizer_frequencies = model.trainer.init_optimizers(model)
    model.trainer.optimizers = optimizers
    model.trainer.lr_schedulers = lr_schedulers
    model.trainer.optimizer_frequencies = optimizer_frequencies
    model.trainer.convert_to_lightning_optimizers()
    model.trainer.is_master = False


def do_nothing_optimizer_closure():
    return


def run_optimizer(ctx, model):
    trainer = model.trainer
    opt_idx = ctx["opt_idx"]
    optimizer = trainer.optimizers[opt_idx]
    closure = getattr(optimizer, "_closure", do_nothing_optimizer_closure)
    optimizer.step(closure=closure)


def save(ctx, model):
    num_gpus_per_model = ctx["num_gpus_per_model"]
    rank = torch_distrib.get_rank()
    if rank in range(num_gpus_per_model):
        seq = list(model.children())[0]
        torch.save(seq, f"seq_{rank}.pt")


def reload_sequential(num_gpus_per_model):
    partial_seqs = [torch.load(f"seq_{rank}.pt", map_location='cpu') for rank in range(num_gpus_per_model)]
    seq = nn.Sequential()
    for p_seq in partial_seqs:
        for name, child in p_seq.named_children():
            seq.add_module(name, child)
    # delete tmp files
    [os.remove(f"seq_{rank}.pt") for rank in range(num_gpus_per_model)]
    return seq


class LightningPipeModule(nn.Module):
    """
        This class wraps Fairscale Pipe and PipeRCPWrapper class.

        Args:
            module: nn.Sequential
                sequential model to be balanced among several gpus

            balance: list of ints
                list of number of layers in each partition.

            checkpoint (str) = 'never'
                when to enable checkpointing, one of ``'always'``,
                ``'except_last'``, or ``'never'`` (default: ``'except_last'``)

            balance_mode: str = "balance_by_size"
                when balance is not provided, the model can be balanced either by size or time.
                refer to balance description.

            mode: PipeMode
                the mode enables switching between Pipe and PipeRCPWrapper class
    """
    def __init__(self,
                 module: nn.Sequential,
                 balance: List[int],
                 microbatches: int = 8,
                 checkpoint='never',
                 pipe_cls=None):
        super().__init__()
        self.module = module
        self.balance = balance
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self._init_pipe(pipe_cls)

    def _init_pipe(self, pipe_cls):
        device = torch.device("cuda", torch_distrib.get_rank())

        self.module = pipe_cls(
            module=self.module,
            balance=self.balance,
            chunks=self.microbatches,
            style=PipelineStyle.MultiProcess,
            input_device=device,
            worker_map=get_worker_map(),
            checkpoint=self.checkpoint,
        )

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return x


class PipeRpcPlugin(RPCPlugin):
    def __init__(self,
                 balance: Optional[List[int]] = None,
                 num_partitions: Optional[int] = None,
                 microbatches: int = 8,
                 checkpoint: str = 'except_last',
                 balance_mode: str = "balance_by_size",
                 pipelined_backward: Optional[bool] = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.balance = balance
        if self.balance is None:
            raise MisconfigurationException(
                'Please, provide a balance for your model. '
                'Example: nn.Sequential(torch.nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 2)) contains 3 layers. '
                'A possible balance between 2 gpus is [2, 1]'
            )

        self.num_partitions = num_partitions
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self.balance_mode = balance_mode
        self.pipelined_backward = pipelined_backward

    @property
    def broadcast_and_barrier_supported(self):
        assert self._broadcast_and_barrier_supported is not None
        return self._broadcast_and_barrier_supported

    def _infering_balance_from_example_input_array(self, trainer):
        model = trainer.get_model()
        if not hasattr(model, "layers") or not isinstance(model.layers, nn.Sequential):
            raise MisconfigurationException(
                'Could not find a PipeLightningModule within the model. '
                'Did you defined set your sequential model as an `layers` attribute of your model ?')

        if self.balance is None:
            partitions = torch.cuda.device_count() if self.num_partitions is None else self.num_partitions
            if model.example_input_array is not None:
                balance_func = getattr(pipe_balance, self.balance_mode)
                self.balance = balance_func(partitions, model.layers, model.example_input_array)
                log.info(f'The following balance {self.balance} was inferended using {self.balance_mode} mode')
            else:
                raise MisconfigurationException(
                    'Please, set example_input_array to your model, so we can infer the right balance for you')

    def _find_pipe_module(self, model):
        # try to wrap for the user
        found_module = False
        if hasattr(model, "layers") and isinstance(model.layers, nn.Sequential):
            model.layers = LightningPipeModule(
                model.layers,
                balance=self.balance,
                microbatches=self.microbatches,
                checkpoint=self.checkpoint,
                pipe_cls=PipeRPCWrapper
            )
            model.final_stage = model.layers.module.final_stage
            model.foreach_worker = model.layers.module.foreach_worker
            model.layers.module.model.trainer = model.trainer
            model.layers.module.model.configure_optimizers = model.configure_optimizers
            found_module = True

        if not found_module:
            raise MisconfigurationException(
                'Could not find a PipeLightningModule within the model. '
                'Did you defined set your sequential model as an `layers` attribute of your model ?')

    def pre_init_ddp_connection(self, trainer) -> bool:
        if torch_distrib.is_initialized():
            return True
        self._infering_balance_from_example_input_array(trainer)
        return False

    def init_rpc_connection_and_pipe(self, trainer, global_rank, world_size):
        os.environ['MASTER_PORT'] = os.getenv('RPC_MASTER_PORT', '15000')
        rpc.init_rpc(f"worker{global_rank}", rank=global_rank, world_size=world_size)

        self.num_gpus_per_model = len(self.balance)
        ensure_divisibility(world_size, self.num_gpus_per_model)
        num_model_parallel = self.num_gpus_per_model / world_size
        mpu.initialize_model_parallel(num_model_parallel, world_size)
        self._broadcast_and_barrier_supported = num_model_parallel > 1

        automatic_optimization = trainer.train_loop.automatic_optimization
        if automatic_optimization:
            raise MisconfigurationException(
                'PipePlugin is currently not supported in automatic optimization')

        if trainer.amp_backend is not None:
            raise MisconfigurationException(
                'PipePlugin is currently not supported in Automatic Mixed Precision')

        self.trainer = trainer
        model = trainer.get_model()

        if global_rank != 0:
            # For RPC, all ranks other than 0 just need to call rpc.shutdown()
            torch_distrib.barrier()
            rpc_pipe.PipeModel.trainer = model.trainer
            rpc_pipe.PipeModel.configure_optimizers = model.configure_optimizers
            torch.distributed.rpc.shutdown()
            return True

        # Create pipe_module
        model = trainer.get_model()
        self._find_pipe_module(model)
        torch_distrib.barrier()
        model.foreach_worker(register_optimizers, include_self=True)
        trainer.is_master = True

    def init_ddp_connection(
            self,
            trainer,
            cluster_environment,
            global_rank: int,
            world_size: int,
            is_slurm_managing_tasks: bool = True,
    ) -> None:

        skip = self.pre_init_ddp_connection(trainer)
        if skip:
            return os.getenv("LOCAL_RANK") != 0

        super().init_ddp_connection(
            trainer=trainer,
            cluster_environment=cluster_environment,
            global_rank=global_rank,
            world_size=world_size,
            is_slurm_managing_tasks=is_slurm_managing_tasks
        )

        return self.init_rpc_connection_and_pipe(trainer, global_rank, world_size)

    def create_optimizers_map(self):
        self._optimizers_map = {opt_idx: False for opt_idx, opt in enumerate(self.trainer.optimizers)}

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        self.create_optimizers_map()
        ddp_plugin = RPCPlugin(process_group=mpu.get_data_parallel_group()).configure_ddp(model, device_ids)
        return ddp_plugin

    def _save_model(self, checkpoint_save_model, last_filepath, trainer, pl_module):
        model = self.trainer.get_model()
        if hasattr(model, "foreach_worker"):
            current_layers = pl_module.layers
            model.foreach_worker(save, {"num_gpus_per_model": self.num_gpus_per_model}, include_self=True)
            pl_module.layers = reload_sequential(self.num_gpus_per_model)
            checkpoint_save_model(last_filepath, trainer, pl_module)
            del pl_module.layers
            pl_module.layers = current_layers

    def _optimizer_step(self, opt_idx, *args, **kwargs):
        # Create pipe_module
        model = self.trainer.get_model()
        model.foreach_worker(run_optimizer, {"opt_idx": opt_idx}, include_self=False)

    def optimizer_step(self, is_master, optimizer, closure, *args, **kwargs):
        if not is_master:
            return False

        opt_idx = optimizer._optimizer_idx
        self._optimizers_map[opt_idx] = not self._optimizers_map[opt_idx]

        if self._optimizers_map[opt_idx]:
            optimizer.step(closure=closure, *args, **kwargs)
            self._optimizer_step(opt_idx, *args, **kwargs)
            return True
        else:
            return False
