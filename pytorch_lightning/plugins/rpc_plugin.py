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

import torch
from torch.distributed import rpc

from pytorch_lightning.plugins.ddp_plugin import DDPPlugin


class RPCPlugin(DDPPlugin):
    """
    Backbone for RPC Plugins built on top of DDP.
    RPC introduces different communication behaviour than DDP. Unlike DDP, processes potentially are not
    required to run the same code as the main process.
    This leads to edge cases where logic needs to be re-defined. This class contains special cases
    that need to be addressed when using RPC communication when building custom RPC Plugins.
    """

    def __init__(self, **kwargs):
        self.rpc_initialized = False
        super().__init__(**kwargs)

    def init_rpc_connection(self,
                            global_rank: int,
                            world_size: int):
        os.environ['MASTER_PORT'] = os.getenv('RPC_MASTER_PORT', '15000')
        rpc.init_rpc(f"worker{global_rank}", rank=global_rank, world_size=world_size)
        self.rpc_initialized = True

    def rpc_save_model(self,
                       save_model_fn,
                       last_filepath,
                       trainer,
                       pl_module):
        raise NotImplementedError

    def on_main_rpc_connection(self, trainer):
        raise NotImplementedError

    def on_exit_rpc_process(self, trainer):
        self.exit_rpc_process()

    def exit_rpc_process(self):
        if self.rpc_initialized:
            torch.distributed.rpc.shutdown()
            self.rpc_initialized = False

    def optimizer_step(self,
                       model,
                       lightning_optimizer,
                       closure,
                       *args,
                       **kwargs):
        raise NotImplementedError

    def is_main_rpc_process(self):
        raise NotImplementedError

    def barrier(self):
        raise NotImplementedError
