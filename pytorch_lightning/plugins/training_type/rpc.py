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
from contextlib import suppress
from typing import List, Optional

import torch

from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment
from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.utilities import _RPC_AVAILABLE

DEFAULT_RPC_TIMEOUT_SEC = 60.
if _RPC_AVAILABLE:
    from torch.distributed import rpc

    with suppress(ModuleNotFoundError, ImportError):
        from torch.distributed.rpc.constants import DEFAULT_RPC_TIMEOUT_SEC


class RPCPlugin(DDPPlugin):
    """
    Backbone for RPC Plugins built on top of DDP.
    RPC introduces different communication behaviour than DDP. Unlike DDP, processes potentially are not
    required to run the same code as the main process.
    This leads to edge cases where logic needs to be re-defined. This class contains special cases
    that need to be addressed when using RPC communication when building custom RPC Plugins.
    """

    def __init__(
        self,
        rpc_timeout_sec: float = DEFAULT_RPC_TIMEOUT_SEC,
        parallel_devices: Optional[List[torch.device]] = None,
        num_nodes: Optional[int] = None,
        cluster_environment: Optional[ClusterEnvironment] = None,
        sync_batchnorm: Optional[bool] = None,
        **kwargs
    ):
        self.rpc_timeout_sec = rpc_timeout_sec
        self._is_rpc_initialized = False
        super().__init__(
            parallel_devices=parallel_devices,
            num_nodes=num_nodes,
            cluster_environment=cluster_environment,
            sync_batchnorm=sync_batchnorm,
            **kwargs
        )

    def init_rpc_connection(self, global_rank: int, world_size: int) -> None:
        os.environ['MASTER_PORT'] = os.getenv('RPC_MASTER_PORT', '15000')
        rpc.init_rpc(f"worker{global_rank}", rank=global_rank, world_size=world_size)
        rpc._set_rpc_timeout(self.rpc_timeout_sec)
        self._is_rpc_initialized = True

    def rpc_save_model(self, save_model_fn, last_filepath, trainer, pl_module) -> None:
        """
        Override to save model to disk.
        This is required as the main process will be required to handle aggregating model states from RPC processes.

        Args:
            save_model_fn: The saving function to save final model.
            last_filepath: The filepath to save the model to.
            trainer: The trainer object.
            pl_module: The LightningModule.
        """
        raise NotImplementedError

    def exit_rpc_process(self):
        if self._is_rpc_initialized:
            torch.distributed.rpc.shutdown()
            self._is_rpc_initialized = False

    @property
    def rpc_enabled(self) -> bool:
        return True
