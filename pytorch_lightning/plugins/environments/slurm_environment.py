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

import logging
import os
import re
from typing import Optional

from pytorch_lightning.plugins.environments.cluster_environment import ClusterEnvironment

log = logging.getLogger(__name__)


class SLURMEnvironment(ClusterEnvironment):
    """Cluster environment for training on a cluster managed by SLURM.

    Args:
        auto_requeue: Whether automatic job resubmission is enabled or not. How and under which conditions a job gets
            rescheduled gets determined by the owner of this plugin.
    """

    def __init__(self, auto_requeue: bool = True) -> None:
        super().__init__()
        self.auto_requeue = auto_requeue

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        # figure out the root node addr
        slurm_nodelist = os.environ.get("SLURM_NODELIST")
        if slurm_nodelist:
            root_node = slurm_nodelist.split(" ")[0].split(",")[0]
        else:
            root_node = "127.0.0.1"

        root_node = self.resolve_root_node_address(root_node)
        os.environ["MASTER_ADDR"] = root_node
        log.debug(f"MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        return root_node

    @property
    def main_port(self) -> int:
        # -----------------------
        # SLURM JOB = PORT number
        # -----------------------
        # this way every process knows what port to use
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is not None:
            # use the last 4 numbers in the job id as the id
            default_port = job_id[-4:]
            # all ports should be in the 10k+ range
            default_port = int(default_port) + 15000
        else:
            default_port = 12910

        # -----------------------
        # PORT NUMBER = MASTER_PORT
        # -----------------------
        # in case the user passed it in
        if "MASTER_PORT" in os.environ:
            default_port = int(os.environ["MASTER_PORT"])
        else:
            os.environ["MASTER_PORT"] = str(default_port)

        log.debug(f"MASTER_PORT: {os.environ['MASTER_PORT']}")
        return default_port

    @staticmethod
    def detect() -> bool:
        """Returns ``True`` if the current process was launched on a SLURM cluster."""
        return "SLURM_NTASKS" in os.environ

    @staticmethod
    def job_name() -> Optional[str]:
        return os.environ.get("SLURM_JOB_NAME")

    @staticmethod
    def job_id() -> Optional[int]:
        # in interactive mode, don't make logs use the same job id
        in_slurm_interactive_mode = SLURMEnvironment.job_name() == "bash"
        if in_slurm_interactive_mode:
            return None

        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is None:
            return None
        try:
            return int(job_id)
        except ValueError:
            return None

    def world_size(self) -> int:
        return int(os.environ["SLURM_NTASKS"])

    def set_world_size(self, size: int) -> None:
        log.debug("SLURMEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        return int(os.environ["SLURM_PROCID"])

    def set_global_rank(self, rank: int) -> None:
        log.debug("SLURMEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        return int(os.environ["SLURM_LOCALID"])

    def node_rank(self) -> int:
        return int(os.environ["SLURM_NODEID"])

    def resolve_root_node_address(self, root_node: str) -> str:
        if "[" in root_node:
            name, numbers = root_node.split("[", maxsplit=1)
            number = numbers.split(",", maxsplit=1)[0]
            if "-" in number:
                number = number.split("-")[0]

            number = re.sub("[^0-9]", "", number)
            root_node = name + number

        return root_node
