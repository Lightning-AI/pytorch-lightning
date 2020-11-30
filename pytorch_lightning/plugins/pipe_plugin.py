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

from pytorch_lightning.utilities import AMPType

try:
    IS_TORCH_AT_LEAST_1_6 = LooseVersion(torch.__version__) >= LooseVersion("1.6.0")
    if IS_TORCH_AT_LEAST_1_6:
        import fairscale.nn.model_parallel as mpu
        from fairscale.nn import Pipe, PipeRPCWrapper
        from fairscale.nn.model_parallel.utils import ensure_divisibility
        from fairscale.nn.pipe import rpc as rpc_pipe
        from fairscale.nn.pipe.pipeline import PipelineStyle
        from torch.distributed import rpc

        # todo: seems to work only for 1.6.0
        HAS_FAIRSCALE = LooseVersion(torch.__version__) == LooseVersion("1.6.0")
    else:
        HAS_FAIRSCALE = False
except Exception as e:
    print(e)
    HAS_FAIRSCALE = False

from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def get_worker_map():
    # TODO, is this correct with multinodes?
    return {rank: f"worker{rank}" for rank in range(torch_distrib.get_world_size())}


def register_optimizers(ctx, model):
    optimizers, lr_schedulers, optimizer_frequencies = model.trainer.init_optimizers(model)
    model.trainer.optimizers = optimizers
    model.trainer.lr_schedulers = lr_schedulers
    model.trainer.optimizer_frequencies = optimizer_frequencies


def do_nothing_optimizer_closure():
    return


def cleanup(ctx, model):
    del model


def run_optimizer(ctx, model):
    trainer = model.trainer
    model_ref = trainer.get_model()
    opt_idx = ctx["opt_idx"]
    args = ctx["args"]
    kwargs = ctx["kwargs"]
    batch_idx = ctx["batch_idx"]
    on_tpu = ctx["on_tpu"]
    optimizer_closure = ctx.pop("optimizer_closure", do_nothing_optimizer_closure)
    optimizer = trainer.optimizers[opt_idx]

    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    using_native_amp = trainer.amp_backend == AMPType.NATIVE
    automatic_optimization = trainer.train_loop.automatic_optimization

    # native amp + lbfgs is a no go right now
    if using_native_amp and is_lbfgs:
        raise MisconfigurationException(
            'native PyTorch amp and lbfgs are not compatible.'
            ' To request, please file a Github issue in PyTorch and tag @mcarilli')

    # model hook
    model_ref.optimizer_step(
        epoch=trainer.current_epoch,
        batch_idx=batch_idx,
        optimizer=optimizer,
        optimizer_idx=opt_idx,
        optimizer_closure=optimizer_closure,
        on_tpu=on_tpu,  # TPUAccelerator class sets this as True
        using_native_amp=using_native_amp,
        using_lbfgs=is_lbfgs,
        *args,
        **kwargs,
    )


class LightningPipeModule(nn.Module):
    def __init__(self, module: nn.Sequential, balance: List[int],
                 microbatches: int = 8, checkpoint='never', version: int = 1, **kwargs):
        super().__init__()
        assert version in [1, 2]
        self._pipe_version = version
        self.module = module
        self.balance = balance
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self._init_pipe()
        self._kwargs = kwargs

    def _init_pipe(self):
        device = torch.device("cuda", torch_distrib.get_rank())
        pipe_cls = Pipe if self._pipe_version == 1 else PipeRPCWrapper
        self.module = pipe_cls(
            module=self.module,
            balance=self.balance,
            chunks=self.microbatches,
            style=PipelineStyle.MultiProcess,
            input_device=device,
            worker_map=get_worker_map(),
            checkpoint=self.checkpoint,
            **self._kwargs
        )

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return x


class PipePlugin(DDPPlugin):
    """This class wraps an arbitrary :class:`nn.Sequential <torch.nn.Sequential>` module
    using Fairscale Pipe_.
    If the module requires lots of memory, Pipe will be very efficient.

    .. _Pipe: https://arxiv.org/abs/1811.06965



    Pipe combines pipeline parallelism with checkpointing to reduce peak
    memory required to train while minimizing device under-utilization.

    You should determine the balance when defining a :class:`Pipe` module, as
    balancing will not be done automatically. The module will be partitioned
    into multiple devices according to the given balance. You may rely on
    heuristics to find your own optimal configuration.

    We expect your LightningModule to contain a sequential under attribute `.layers`

    ::
        class Model(LightningModule):

            ....

            self.layers = nn.Sequential(...)

    Args:
        balance (ints):
            list of number of layers in each partition

        microbatches: int = 8:
            batches bigger than microbatches will be splitted to keep memory constrained

        checkpoint (str) = 'except_last'
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)

        version: int = 2
            Lightning supports both Fairscale Pipe (v1) and PipeRPCWrapper (v2) implementations

        kwarg: Any

    """
    def __init__(self, balance: List[int], microbatches: int = 8, checkpoint='never', version: int = 1, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(balance, list) and len(balance) > 0
        self.balance = balance
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self.version = version
        self._use_barrier_and_broadcast = None
        self._kwargs = kwargs

    @property
    def use_barrier_and_broadcast(self):
        return self._use_barrier_and_broadcast

    @property
    def use_optimizer_step(self):
        return self.version != 1

    def _find_pipe_module(self, model):
        # try to wrap for the user
        if hasattr(model, "layers") and isinstance(model.layers, nn.Sequential):
            model.layers = LightningPipeModule(
                model.layers,
                balance=self.balance,
                microbatches=self.microbatches,
                checkpoint=self.checkpoint,
                version=self.version,
                **self._kwargs
            )
            model.final_stage = model.layers.module.final_stage
            if self.version == 1:
                model.back_helper = model.layers.module.back_helper
            else:
                model.foreach_worker = model.layers.module.foreach_worker
                model.layers.module.model.trainer = model.trainer
                model.layers.module.model.configure_optimizers = model.configure_optimizers
            pipe_module = model
            found_module = True

        if not found_module:
            raise MisconfigurationException(
                'Could not find a PipeLightningModule within the model. '
                'Did you defined set your sequential model as an `layers` attribute of your model ?')
        return pipe_module

    def init_ddp_connection(
            self,
            trainer,
            cluster_environment,
            global_rank: int,
            world_size: int,
            is_slurm_managing_tasks: bool = True,
    ) -> None:
        super().init_ddp_connection(
            trainer=trainer,
            cluster_environment=cluster_environment,
            global_rank=global_rank,
            world_size=world_size,
            is_slurm_managing_tasks=is_slurm_managing_tasks
        )
        os.environ["MASTER_PORT"] = "15000"
        rpc.init_rpc(f"worker{global_rank}", rank=global_rank, world_size=world_size)

        num_gpus_per_model = len(self.balance)
        ensure_divisibility(world_size, num_gpus_per_model)
        num_model_parallel = num_gpus_per_model / world_size
        mpu.initialize_model_parallel(num_model_parallel, world_size)
        self._use_barrier_and_broadcast = (self.version == 1 or num_model_parallel > 1)

        automatic_optimization = trainer.train_loop.automatic_optimization
        if automatic_optimization:
            raise MisconfigurationException(
                'PipePlugin is currently not supported in automatic optimization')

        if trainer.amp_backend is not None:
            raise MisconfigurationException(
                'PipePlugin is currently not supported in Automatic Mixed Precision')

        self.trainer = trainer
        model = trainer.get_model()

        if self.version == 2:
            if global_rank != 0:
                # For RPC, all ranks other than 0 just need to call rpc.shutdown()
                torch_distrib.barrier()
                rpc_pipe.PipeModel.trainer = model.trainer
                rpc_pipe.PipeModel.configure_optimizers = model.configure_optimizers
                torch.distributed.rpc.shutdown()
                return

        # Create pipe_module
        model = trainer.get_model()
        self._find_pipe_module(model)
        if self.version == 2:
            torch_distrib.barrier()
            model.foreach_worker(register_optimizers, include_self=True)

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:

        model.trainer.init_optimizers(model)
        return DDPPlugin(process_group=mpu.get_data_parallel_group()).configure_ddp(model, device_ids)

    def optimizer_step(self, optimizer, batch_idx, opt_idx, optimizer_closure, on_tpu, *args, **kwargs):
        # Create pipe_module
        automatic_optimization = self.trainer.train_loop.automatic_optimization
        model = self.trainer.get_model()
        ctx = {"batch_idx":batch_idx, "opt_idx": opt_idx, "on_tpu": on_tpu, "args": args, "kwargs":kwargs, "optimizer_closure": optimizer_closure}
        if not automatic_optimization:
            run_optimizer(ctx, model)
            model.foreach_worker(run_optimizer, ctx, include_self=False)
        else:
            raise NotImplementedError
