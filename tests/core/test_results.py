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
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tests.helpers.utils as tutils
from pytorch_lightning.trainer.connectors.logger_connector.result import _Sync
from pytorch_lightning.utilities.distributed import sync_ddp_if_available
from tests.helpers.runif import RunIf


def _setup_ddp(rank, worldsize):
    import os

    os.environ["MASTER_ADDR"] = "localhost"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def _ddp_test_fn(rank, worldsize):
    _setup_ddp(rank, worldsize)
    tensor = torch.tensor([1.0])
    sync = _Sync(sync_ddp_if_available, _should=True, op="SUM")
    actual = sync(tensor)
    assert actual.item() == dist.get_world_size(), "Result-Log does not work properly with DDP and Tensors"


@RunIf(skip_windows=True)
def test_result_reduce_ddp():
    """Make sure result logging works with DDP."""
    tutils.set_random_master_port()
    worldsize = 2
    mp.spawn(_ddp_test_fn, args=(worldsize,), nprocs=worldsize)
