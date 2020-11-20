import os
from enum import Enum
from typing import List, Optional

import fairscale.nn.model_parallel as mpu
import numpy as np
import torch
import torch.distributed as torch_distrib
from fairscale.nn.pipe.pipeline import PipelineStyle
from torch import nn
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(1200, 19000, 1))


def get_random_port():
    seed_everything(np.random.randint(1, 10))
    return str(RANDOM_PORTS.pop())


def get_worker_map():
    # TODO, is this correct with multinodes?
    return {rank: f"worker{rank}" for rank in range(torch_distrib.get_world_size())}


class LightningPipeModule(nn.Module):

    def __init__(self, module: nn.Sequential, balance: List[int], microbatches: int = 8, checkpoint='never', version: int = 1):
        super().__init__()
        assert version in [1, 2]
        self._pipe_version = version
        self.module = module
        self.balance = balance
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self._init_pipe()

    def _init_pipe(self):
        device = torch.device("cuda", torch_distrib.get_rank())
        if self._pipe_version == 1:
            from fairscale.nn import Pipe
            self.module = Pipe(
                module=self.module,
                balance=self.balance,
                chunks=self.microbatches,
                style=PipelineStyle.MultiProcess,
                input_device=device,
                worker_map=get_worker_map(),
                checkpoint=self.checkpoint)
        else:
            from fairscale.nn import PipeRPCWrapper
            self.module = PipeRPCWrapper(
                module=self.module,
                balance=self.balance,
                chunks=self.microbatches,
                style=PipelineStyle.MultiProcess,
                input_device=device,
                worker_map=get_worker_map(),
                checkpoint=self.checkpoint,
            )

    @property
    def final_stage(self):
        return self.module.final_stage

    @property
    def back_helper(self):
        return self.module.back_helper

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return x


class ModelParallelPlugin(DDPPlugin):
    def __init__(self, balance: List[int], microbatches: int = 8, checkpoint='never', version: int = 1, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(balance, list) and len(balance) > 0
        self.balance = balance
        self.microbatches = microbatches
        self.checkpoint = checkpoint
        self.version = version

    def on_setup(self, model):
        # TODO this should be connected to the accelerators via a hook, hasn't been yet...
        self.pipe_module = self._find_pipe_module(model)

    def _find_pipe_module(self, model):
        pipe_module = None
        found_module = False
        for m in model.modules():
            if type(m) is LightningPipeModule:
                pipe_module = m
                if found_module:
                    raise MisconfigurationException('Currently DDP Pipe only supports one PipeLightningModule')
                found_module = True

        # try to wrap for the user
        if not found_module and hasattr(model, "layers") and isinstance(model.layers, nn.Sequential):
            model.layers = LightningPipeModule(
                model.layers,
                balance=self.balance,
                microbatches=self.microbatches,
                checkpoint=self.checkpoint,
                version=self.version
            )
            model.final_stage = model.layers.final_stage
            model.back_helper = model.layers.back_helper
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

        os.environ["MASTER_PORT"] = get_random_port()  # TODO change...
        rpc.init_rpc(f"worker{global_rank}", rank=global_rank, world_size=world_size)
        mpu.initialize_model_parallel(1, world_size)

        # Create pipe_module
        model_ref = trainer.get_model()
        self.pipe_module = self._find_pipe_module(model_ref)

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        self.ddp_plugin = DDPPlugin(process_group=mpu.get_data_parallel_group())
        model = self.ddp_plugin.configure_ddp(model, device_ids)
        return model
