# Copyright The Lightning AI team.
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
import shutil
import signal
import sys
from typing import Optional

from lightning_fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.rank_zero import rank_zero_warn
from lightning_fabric.utilities.warnings import PossibleUserWarning

log = logging.getLogger(__name__)


class SLURMEnvironment(ClusterEnvironment):
    """Cluster environment for training on a cluster managed by SLURM.

    Args:
        auto_requeue: Whether automatic job resubmission is enabled or not. How and under which conditions a job gets
            rescheduled gets determined by the owner of this plugin.
        requeue_signal: The signal that SLURM will send to indicate that the job should be requeued. Defaults to
            SIGUSR1 on Unix.
    """

    def __init__(self, auto_requeue: bool = True, requeue_signal: Optional[signal.Signals] = None) -> None:
        super().__init__()
        self.auto_requeue = auto_requeue
        if requeue_signal is None and not _IS_WINDOWS:
            requeue_signal = signal.SIGUSR1
        self.requeue_signal = requeue_signal
        self._validate_srun_used()
        self._validate_srun_variables()

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        nodelist = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        root_node = self.resolve_root_node_address(nodelist)
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
        """Returns ``True`` if the current process was launched on a SLURM cluster.

        It is possible to use the SLURM scheduler to request resources and then launch processes manually using a
        different environment. For this, the user can set the job name in SLURM to 'bash' (``SLURM_JOB_NAME=bash``).
        This will then avoid the detection of ``SLURMEnvironment`` and another environment can be detected
        automatically.
        """
        SLURMEnvironment._validate_srun_used()
        return _is_srun_used()

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

    @staticmethod
    def resolve_root_node_address(nodes: str) -> str:
        """The node selection format in SLURM supports several formats.

        This function selects the first host name from

        - a space-separated list of host names, e.g., 'host0 host1 host3' yields 'host0' as the root
        - a comma-separated list of host names, e.g., 'host0,host1,host3' yields 'host0' as the root
        - the range notation with brackets, e.g., 'host[5-9]' yields 'host5' as the root
        """
        nodes = re.sub(r"\[(.*?)[,-].*\]", "\\1", nodes)  # Take the first node of every node range
        nodes = re.sub(r"\[(.*?)\]", "\\1", nodes)  # handle special case where node range is single number
        return nodes.split(" ")[0].split(",")[0]

    @staticmethod
    def _validate_srun_used() -> None:
        """Checks if the `srun` command is available and used.

        Parallel jobs (multi-GPU, multi-node) in SLURM are launched by prepending `srun` in front of the Python command.
        Not doing so will result in processes hanging, which is a frequent user error. Lightning will emit a warning if
        `srun` is found but not used.
        """
        if _IS_WINDOWS:
            return

        srun_exists = shutil.which("srun") is not None
        if srun_exists and not _is_srun_used():
            hint = " ".join(["srun", os.path.basename(sys.executable), *sys.argv])[:64]
            rank_zero_warn(
                "The `srun` command is available on your system but is not used. HINT: If your intention is to run"
                f" Lightning on SLURM, prepend your python command with `srun` like so: {hint} ...",
                category=PossibleUserWarning,
            )

    @staticmethod
    def _validate_srun_variables() -> None:
        """Checks for conflicting or incorrectly set variables set through `srun` and raises a useful error
        message.

        Right now, we only check for the most common user errors. See `the srun docs
        <https://slurm.schedmd.com/srun.html>`_ for a complete list of supported srun variables.
        """
        ntasks = int(os.environ.get("SLURM_NTASKS", "1"))
        if ntasks > 1 and "SLURM_NTASKS_PER_NODE" not in os.environ:
            raise RuntimeError(
                f"You set `--ntasks={ntasks}` in your SLURM bash script, but this variable is not supported."
                f" HINT: Use `--ntasks-per-node={ntasks}` instead."
            )


def _is_srun_used() -> bool:
    return "SLURM_NTASKS" in os.environ and SLURMEnvironment.job_name() != "bash"
