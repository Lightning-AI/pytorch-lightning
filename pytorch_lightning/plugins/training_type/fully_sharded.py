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
from typing import List, Optional

import torch

from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities import _FAIRSCALE_FULLY_SHARDED_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _FAIRSCALE_FULLY_SHARDED_AVAILABLE:
    from fairscale.nn.data_parallel import FullyShardedDataParallel

    from pytorch_lightning.overrides.fairscale import (
        LightningFullyShardedDataParallel,
        unwrap_lightning_module_fully_sharded,
    )


class FullyShardedPlugin(DDPPlugin):

    def __init__(
        self,
        cpu_offload: bool = True,
        flatten_parameters: bool = False,
        reshard_after_forward: bool = True,
        move_grads_to_cpu: Optional[bool] = None,
        fp32_reduce_scatter: Optional[bool] = None,
        compute_dtype: Optional[torch.dtype] = None,
        bucket_cap_mb: int = 25,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: int = 1,
        cluster_environment: ClusterEnvironment = None,
        sync_batchnorm: Optional[bool] = False
    ):
        """

        Provides capabilities to run training using the Full Sharded capabilities provided by FairScale.

        Full Sharded Training shards the entire model across all available GPUs, allowing you to scale model
        size, whilst using efficient communication to reduce overhead. In practice, this means we can remain
        at parity with PyTorch DDP, whilst scaling our model sizes dramatically. The technique is similar
        to ZeRO-Stage 3 but have been modified/adjusted for PyTorch.

        `For more information: https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html`.

        .. warning:: ``FullyShardedPlugin`` is in beta and subject to change.

        Defaults have been set to enable CPU Offload, but options have been exposed and may require configuration
        based on your level of memory/speed efficiency.
        We suggest having a look at this PR for more information.
        `https://github.com/facebookresearch/fairscale/pull/413`


        Many of the helpful doc strings below came from the original FairScale documentation:
        `https://fairscale.readthedocs.io/en/latest/api/nn/fsdp.html`

        Arguments:

           cpu_offload: Offload FP32 params to CPU. Only useable in precision=16 mode (default: False).

           move_grads_to_cpu: Moves gradient shards to CPU after reducation.
                Only disable if using CPU based optimizers (defaults to ``cpu_offload``).

           flatten_parameters: Flattens parameter into single contiguous tensor for speed efficiency
                (default: False).

           reshard_after_forward: Reshard parameters after the forward pass, which saves memory but slows
                down training. Only revelant when nesting FullyShardedDataParallel wrappers inside the model.
                (default: False).

           fp32_reduce_scatter: Reduce-Scatter gradients in FP32. Only relevant in mixed precision
                (default: None)

           compute_dtype: dtype for full parameters for computation. Default to torch.float32,
                unless using mixed precision, in which case defaults to torch.float16.

           bucket_cap_mb: bucket parameters so that gradient reduction
           can potentially overlap with backward computation.
           bucket_cap_mb controls the bucket size in MegaBytes (MB).
           Buckets are sub-divided based on world_size,
           so the max shard size is roughly bucket_cap_mb / world_size.
           Values <= 0 disable bucketing. (Default: 25).

        """
        if not _FAIRSCALE_FULLY_SHARDED_AVAILABLE:
            raise MisconfigurationException(
                "Full Sharded Training is not available. Install the latest FairScale via `pip install fairscale -U`"
            )

        if sync_batchnorm:
            raise MisconfigurationException("Currently sync batch norm is not supported by Full Sharded Training.")
        super().__init__(parallel_devices, num_nodes, cluster_environment, sync_batchnorm=sync_batchnorm)
        self.cpu_offload = cpu_offload
        self.move_grads_to_cpu = move_grads_to_cpu
        self.flatten_parameters = flatten_parameters
        self.reshard_after_forward = reshard_after_forward
        self.fp32_reduce_scatter = fp32_reduce_scatter
        self.compute_dtype = compute_dtype
        self.bucket_cap_mb = bucket_cap_mb

    def configure_ddp(self):
        precision = self.lightning_module.trainer.precision
        self.model = FullyShardedDataParallel(
            LightningFullyShardedDataParallel(self.model),
            cpu_offload=self.cpu_offload,
            move_grads_to_cpu=self.move_grads_to_cpu,
            flatten_parameters=self.flatten_parameters,
            mixed_precision=precision == "mixed",
            reshard_after_forward=self.reshard_after_forward,
            fp32_reduce_scatter=self.fp32_reduce_scatter,
            compute_dtype=self.compute_dtype,
            bucket_cap_mb=self.bucket_cap_mb,
        )

    @property
    def lightning_module(self) -> LightningModule:
        return unwrap_lightning_module_fully_sharded(self.model)

    def model_to_device(self):
        if not self.cpu_offload:
            super().model_to_device()

    def on_save(self, checkpoint: dict) -> dict:
        state_dict = self.collate_state_dict()
        checkpoint['state_dict'] = state_dict
        return checkpoint

    def collate_state_dict(self):
        """
        Collects the models sharded state dict from all processes before returning.
        Returns: The unsharded model state dict.
        """
        state_dict = self.model.state_dict()
        # Remove module prefix from state dict as this is the behaviour of state dict.
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        return state_dict
