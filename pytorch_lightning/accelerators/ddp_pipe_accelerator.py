import os
from typing import List

import fairscale.nn.model_parallel as mpu
import torch
import torch.distributed as torch_distrib
from fairscale.nn import Pipe
from fairscale.nn.pipe.pipeline import PipelineStyle
from torch import nn
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import LightningModule
from pytorch_lightning import _logger as log
from pytorch_lightning.accelerators import DDPAccelerator
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def get_worker_map():
    return {rank: f"Test{rank}" for rank in range(torch_distrib.get_world_size())}


class LightningPipeModule(nn.Module):
    def __init__(self, module: nn.Sequential, layer_partitions: List[int], microbatches: int = 8):
        super().__init__()
        self.module = module
        self.layer_partitions = layer_partitions
        self.microbatches = microbatches

    def init_pipe(self):
        self.module = Pipe(
            module=self.module,
            balance=self.layer_partitions,
            chunks=self.microbatches,
            style=PipelineStyle.MultiProcess,
            input_device=torch.cuda.current_device(),
            worker_map=get_worker_map(),
            checkpoint='never',
        )

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return x


class DDPPipeAccelerator(DDPAccelerator):
    def __init__(self, trainer=None, cluster_environment=None, ddp_plugin=None):
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.nickname = 'ddp_pipe'
        self.pipe_module = None  # Initialized at model setup

    def init_ddp_connection(
            self, global_rank: int, world_size: int, is_slurm_managing_tasks: bool = True
    ) -> None:
        os.environ["MASTER_ADDR"] = str(self.cluster_environment.master_address())
        os.environ["MASTER_PORT"] = str(self.cluster_environment.master_port())
        os.environ["WORLD_SIZE"] = str(self.cluster_environment.world_size())

        torch_backend = "nccl" if self.trainer.on_gpu else "gloo"

        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

        if not torch.distributed.is_initialized():
            log.info(
                f"initializing ddp: GLOBAL_RANK: {global_rank}, MEMBER: {global_rank + 1}/{world_size}"
            )
            torch_distrib.init_process_group(
                torch_backend, rank=global_rank, world_size=world_size, init_method=init_method
            )

        os.environ["MASTER_PORT"] = "10639"  # TODO change...
        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

        rpc.init_rpc(
            f"Test{global_rank}",
            rank=global_rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
        )
        mpu.initialize_model_parallel(model_parallel_size_=1, pipeline_length=len(self.pipe_module.layer_partitions))

    def model_to_device(self, model, process_idx):
        self.trainer.root_gpu = self.trainer.data_parallel_device_ids[self.trainer.local_rank]
        torch.cuda.set_device(self.trainer.root_gpu)
        self.pipe_module = self._find_pipe_module(model)
        self.pipe_module.init_pipe()
        model.cuda(self.trainer.root_gpu)

    def _find_pipe_module(self, model):
        pipe_module = None
        found_module = False
        for m in model.modules():
            if type(m) is LightningPipeModule:
                pipe_module = m
                if found_module:
                    raise MisconfigurationException('Currently DDP Pipe only supports one PipeLightningModule')
                found_module = True
        if not found_module:
            raise MisconfigurationException(
                'Could not find a PipeLightningModule within the model. '
                'Did you wrap your sequential model with the PipeLightningModule class?')
        return pipe_module

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        self.ddp_plugin = DDPPlugin(process_group=mpu.get_data_parallel_group())
        model = self.ddp_plugin.configure_ddp(model, device_ids)
        return model

    @property
    def final_stage(self):
        return self.pipe_module.module.final_stage

    def sync_gradients(self, output):
        self.pipe_module.module.back_helper(output)
