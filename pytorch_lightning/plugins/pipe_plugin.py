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
import os
from distutils.version import LooseVersion
from typing import List, Optional

import torch
import torch.distributed as torch_distrib
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import LightningModule
from pytorch_lightning import _logger as log
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.utilities import FAIRSCALE_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

FAIRSCALE_AVAILABLE &= LooseVersion(torch.__version__) == LooseVersion("1.6.0")

if FAIRSCALE_AVAILABLE:
    import fairscale.nn.model_parallel as mpu
    from fairscale.nn import LazyModule, Pipe
    from fairscale.nn.model_parallel.utils import ensure_divisibility
    from fairscale.nn.pipe import balance as pipe_balance
    from fairscale.nn.pipe.pipeline import PipelineStyle
    from torch.distributed import rpc


def get_worker_map():
    # TODO, is this correct with multinodes?
    return {rank: f"worker{rank}" for rank in range(torch_distrib.get_world_size())}


def to_lazy(layer):
    return LazyModule(lambda: layer)


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
                 enable_lazy_module=True,
                 pipe_cls=None):
        super().__init__()
        self.module = module
        self.balance = balance
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self.enable_lazy_module = enable_lazy_module
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

        # del self.module.model.mp_partitions
        # torch.cuda.empty_cache()

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
        balance Optional (ints):
            list of number of layers in each partition. When not provided,
            it will use `example_input_array`, it will make an inference
            while recording

        microbatches: int = 8:
            batches bigger than microbatches will be splitted to keep memory constrained

        checkpoint (str) = 'except_last'
            when to enable checkpointing, one of ``'always'``,
            ``'except_last'``, or ``'never'`` (default: ``'except_last'``)

        balance_mode: str = "balance_by_size"
            when balance is not provided, the model can be balanced either by size or time.
            refer to balance description.

        pipelined_backward (bool, optional):
            if True, call torch.autograd.backward once per microbatch on the
            backward pass (instead of once for the whole batch). This works
            around a potential deadlock in pytorch when using tensor parallelism
            at the same time. Defaults to `True` if
            `get_model_parallel_world_size() > 1`
            (default: `None`)

        kwarg: Any

    """
    def __init__(self,
                 balance: Optional[List[int]] = None,
                 num_partitions: Optional[int] = None,
                 microbatches: int = 8,
                 checkpoint: str = 'except_last',
                 balance_mode: str = "balance_by_size",
                 pipelined_backward: Optional[bool] = None,
                 **kwargs):
        super().__init__(**kwargs)

        self.balance = balance
        self.num_partitions = num_partitions
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self.balance_mode = balance_mode
        self.pipelined_backward = pipelined_backward

        if self.balance_mode not in ["balance_by_time", "balance_by_size"]:
            raise MisconfigurationException(
                'balance_mode can either be balance_by_time or balance_by_size')

        self._kwargs = kwargs

    @property
    def broadcast_and_barrier_supported(self):
        return True

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
        if hasattr(model, "layers") and isinstance(model.layers, nn.Sequential):
            model.layers = LightningPipeModule(
                model.layers,
                balance=self.balance,
                microbatches=self.microbatches,
                checkpoint=self.checkpoint,
                pipe_cls=Pipe
            )
            model.final_stage = model.layers.module.final_stage
            model.back_helper = model.layers.module.back_helper
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

        self._infering_balance_from_example_input_array(trainer)

        num_gpus_per_model = len(self.balance)
        ensure_divisibility(world_size, num_gpus_per_model)
        num_model_parallel = num_gpus_per_model / world_size
        mpu.initialize_model_parallel(num_model_parallel, world_size)

        automatic_optimization = trainer.train_loop.automatic_optimization
        if automatic_optimization:
            raise MisconfigurationException(
                'PipePlugin is currently not supported in automatic optimization')

        if trainer.amp_backend is not None:
            raise MisconfigurationException(
                'PipePlugin is currently not supported in Automatic Mixed Precision')

        self.trainer = trainer
        model = trainer.get_model()

        # Create pipe_module
        model = trainer.get_model()
        self._find_pipe_module(model)

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:

        model.trainer.init_optimizers(model)
        return DDPPlugin(process_group=mpu.get_data_parallel_group()).configure_ddp(model, device_ids)
