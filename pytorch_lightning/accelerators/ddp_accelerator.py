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
import os
import subprocess
import sys
from os.path import abspath
from time import sleep
from typing import Optional

import numpy as np

from pytorch_lightning.accelerators.distributed_accelerator import DistributedAccelerator
from pytorch_lightning.distributed.dist import LightningDistributed
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.utilities.distributed import find_free_network_port
from pytorch_lightning.utilities.exceptions import MisconfigurationException

try:
    from hydra.utils import to_absolute_path, get_original_cwd
    from hydra.core.hydra_config import HydraConfig
except ImportError:
    HYDRA_AVAILABLE = False
else:
    HYDRA_AVAILABLE = True


class DDPAccelerator(DistributedAccelerator):

    def __init__(self, trainer, cluster_environment=None, ddp_plugin=None):
        """
        Runs training using DDP strategy on a single machine (manually, not via cluster start)

        Example::

            # default
            trainer = Trainer(accelerator=DDPAccelerator())

        """
        super().__init__(trainer, cluster_environment, ddp_plugin)
        self.task_idx = None
        self._has_spawned_children = False
        self.interactive_ddp_procs = []
        self.dist = LightningDistributed()
        self.nickname = 'ddp'

    def setup(self, model):
        # first track model
        self.trainer.model = model

        # start the other scripts
        if os.environ.get('PL_IN_DDP_SUBPROCESS', '0') != '1':
            self._call_children_scripts()

        # set the task idx
        self.task_idx = int(os.environ['LOCAL_RANK'])

    def _call_children_scripts(self):
        assert self.trainer.global_rank == 0
        self._check_can_spawn_children()
        self._has_spawned_children = True

        os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', '127.0.0.1')
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', str(find_free_network_port()))

        # allow the user to pass the node rank
        node_rank = '0'
        node_rank = os.environ.get('NODE_RANK', node_rank)
        node_rank = os.environ.get('GROUP_RANK', node_rank)
        os.environ['NODE_RANK'] = node_rank
        os.environ['LOCAL_RANK'] = '0'

        # when user is using hydra find the absolute path
        path_lib = abspath if not HYDRA_AVAILABLE else to_absolute_path

        # pull out the commands used to run the script and resolve the abs file path
        command = sys.argv
        try:
            full_path = path_lib(command[0])
        except Exception as e:
            full_path = abspath(command[0])

        command[0] = full_path
        # use the same python interpreter and actually running
        command = [sys.executable] + command

        # the visible devices tell us how many GPUs we want to use.
        # when the trainer script was called the device has already been scoped by the time
        # code reaches this point. so, to call the scripts, we need to leave cuda visible devices alone
        # but forward the GPUs selected via environment variables
        if self.trainer.data_parallel_device_ids is None:
            raise MisconfigurationException('you selected (distribute_backend = ddp) but did not set Trainer(gpus=?)')

        os.environ['PL_TRAINER_GPUS'] = ','.join([str(i) for i in self.trainer.data_parallel_device_ids])
        os.environ['PL_IN_DDP_SUBPROCESS'] = '1'

        if self.trainer.logger is not None:
            os.environ['PL_EXP_VERSION'] = str(self.trainer.logger.version)

        num_gpus = len(self.trainer.data_parallel_device_ids)
        os.environ['WORLD_SIZE'] = f'{num_gpus * self.trainer.num_nodes}'

        self.interactive_ddp_procs = []
        for local_rank in range(1, self.trainer.num_processes):
            env_copy = os.environ.copy()
            env_copy['LOCAL_RANK'] = f'{local_rank}'

            # remove env var if global seed not set
            if os.environ.get('PL_GLOBAL_SEED') is None and 'PL_GLOBAL_SEED' in env_copy:
                del env_copy['PL_GLOBAL_SEED']

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            cwd: Optional[str] = None
            if HYDRA_AVAILABLE:
                if HydraConfig.initialized():
                    cwd = get_original_cwd()
            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.interactive_ddp_procs.append(proc)

            # starting all processes at once can cause issues
            # with dataloaders delay between 1-10 seconds
            delay = np.random.uniform(1, 5, 1)[0]
            sleep(delay)

    def train(self):
        model = self.trainer.model

        results = self.ddp_train(process_idx=self.task_idx, model=model)
        if 'WORLD_SIZE' in os.environ:
            del os.environ['WORLD_SIZE']
        return results

    def _check_can_spawn_children(self):
        if self._has_spawned_children:
            raise RuntimeError(
                "You tried to run `.fit` or `.test` multiple times in the same script."
                " This is not supported in DDP mode, switch to `distributed_backend='ddp_spawn'` instead."
            )

    def ddp_train(self, process_idx, model):
        seed = os.environ.get("PL_GLOBAL_SEED")
        if seed is not None:
            seed_everything(int(seed))
        return super().ddp_train(process_idx, model)
