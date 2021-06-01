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
from abc import ABC, abstractmethod
from typing import Dict
from pytorch_lightning.utilities import rank_zero_warn
import os


class ClusterEnvironment(ABC):
    """ Specification of a cluster environment. """

    DEFAULT_ENVIRON_SETTINGS = {
        # these are default NCCL settings for communication speedup
        "NCCL_NSOCKS_PERTHREA": "4",
        "NCCL_SOCKET_NTHREADS": "2",
    }

    def __init__(self, environ_settings: Dict[str, str] = {}):
        # setting default environment if not os.environ not set.
        for environ_param, value in self.DEFAULT_ENVIRON_SETTINGS.items():
            if environ_param in os.environ:
                rank_zero_warn(
                    f"environ parameter {environ_param}: {os.environ.get(environ_param)} "
                    "is already set, will not apply default value."
                )
            else:
                os.environ[environ_param] = value
                rank_zero_warn(
                    f"Setting environ parameter {environ_param} to default value: {value}."
                )
        # override os.environ from user defined `environ_settings`
        for environ_param, value in environ_settings.items():
            if environ_param in os.environ:
                rank_zero_warn(
                    f"environ parameter {environ_param}: {os.environ.get(environ_param)} "
                    f"will be overriden to user defined new value: {value}."
                )
            os.environ[environ_param] = value

    @abstractmethod
    def creates_children(self) -> bool:
        """ Whether the environment creates the subprocesses or not. """

    @abstractmethod
    def master_address(self) -> str:
        """ The master address through which all processes connect and communicate. """

    @abstractmethod
    def master_port(self) -> int:
        """ An open and configured port in the master node through which all processes communicate. """

    @abstractmethod
    def world_size(self) -> int:
        """ The number of processes across all devices and nodes. """

    @abstractmethod
    def set_world_size(self, size: int) -> None:
        pass

    @abstractmethod
    def global_rank(self) -> int:
        """ The rank (index) of the currently running process across all nodes and devices. """

    @abstractmethod
    def set_global_rank(self, rank: int) -> None:
        pass

    @abstractmethod
    def local_rank(self) -> int:
        """ The rank (index) of the currently running process inside of the current node. """

    @abstractmethod
    def node_rank(self) -> int:
        """ The rank (index) of the node on which the current process runs. """

    def teardown(self) -> None:
        """ Clean up any state set after execution finishes. """
        pass
