import os
from typing import List

import fairscale.nn.model_parallel as mpu
import torch
import torch.distributed as torch_distrib
from fairscale.nn import PipeRPCWrapper
from fairscale.nn.pipe.pipeline import PipelineStyle
from torch import nn
from torch.distributed import rpc
from torch.nn.parallel import DistributedDataParallel

from pytorch_lightning import LightningModule
from pytorch_lightning.plugins.ddp_plugin import DDPPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException


def get_worker_map():
    # TODO, is this correct with multinodes?
    return {rank: f"Test{rank}" for rank in range(torch_distrib.get_world_size())}


class LightningPipeModule(nn.Module):
    def __init__(self, module: nn.Sequential, balance: List[int], microbatches: int = 8, checkpoint='never'):
        super().__init__()
        self.module = module
        self.balance = balance
        self.microbatches = microbatches
        self.checkpoint = checkpoint

    def init_pipe(self):
        self.module = PipeRPCWrapper(
            module=self.module,
            balance=self.balance,
            chunks=self.microbatches,
            style=PipelineStyle.MultiProcess,
            input_device=torch.cuda.current_device(),
            worker_map=get_worker_map(),
            checkpoint=self.checkpoint,
        )

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return x


class PipePlugin(DDPPlugin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        if not found_module:
            raise MisconfigurationException(
                'Could not find a PipeLightningModule within the model. '
                'Did you wrap your sequential model with the PipeLightningModule class?')
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

        os.environ["MASTER_PORT"] = "10639"  # TODO change...
        init_method = f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"

        rpc.init_rpc(
            f"Test{global_rank}",
            rank=global_rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=rpc.TensorPipeRpcBackendOptions(init_method=init_method),
        )
        mpu.initialize_model_parallel(model_parallel_size_=1, pipeline_length=len(self.pipe_module.balance))

        if self._not_control_rank(global_rank):
            # For RPC, all ranks other than 0 (??) just need to call rpc.shutdown()
            torch.distributed.rpc.shutdown()
            return

    def _not_control_rank(self, rank):
        # TODO I think this is wrong, what if we have multiple data parallel groups?
        return rank != 0

    def configure_ddp(
            self, model: LightningModule, device_ids: List[int]
    ) -> DistributedDataParallel:
        self.pipe_module.init_pipe()
        self.ddp_plugin = DDPPlugin(process_group=mpu.get_data_parallel_group())
        model = self.ddp_plugin.configure_ddp(model, device_ids)
        return model
