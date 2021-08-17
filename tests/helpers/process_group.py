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
from contextlib import contextmanager

import torch.distributed as dist

from pytorch_lightning.plugins.environments.lightning_environment import find_free_network_port


@contextmanager
def single_process_pg():
    """
    Initialize the default process group with only the current process for
    testing purposes. The process group is destroyed when the with block is
    exited.
    """
    if dist.is_initialized():
        raise RuntimeError("Can't use `single_process_pg ` when the default process group is already initialized.")

    orig_environ = os.environ.copy()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_free_network_port())
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    dist.init_process_group("gloo")
    try:
        yield
    finally:
        dist.destroy_process_group()
        os.environ.clear()
        os.environ.update(orig_environ)
