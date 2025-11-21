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
import shutil
import signal
import sys
from typing import Optional

from typing_extensions import override

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.utilities.imports import _IS_WINDOWS, _raise_enterprise_not_available
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.fabric.utilities.warnings import PossibleUserWarning

log = logging.getLogger(__name__)


class SLURMEnvironment(ClusterEnvironment):
    """Cluster environment for training on a cluster managed by SLURM.

    You can configure the `main_address` and `main_port` properties via the env variables `MASTER_ADDR` and
    `MASTER_PORT`, respectively.

    Args:
        auto_requeue: Whether automatic job resubmission is enabled or not. How and under which conditions a job gets
            rescheduled gets determined by the owner of this plugin.
        requeue_signal: The signal that SLURM will send to indicate that the job should be requeued. Defaults to
            SIGUSR1 on Unix.

    """

    def __init__(self, auto_requeue: bool = True, requeue_signal: Optional[signal.Signals] = None) -> None:
        super().__init__()
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.environments.slurm import (
            SLURMEnvironment as EnterpriseSLURMEnvironment,
        )

        self.slurm_impl = EnterpriseSLURMEnvironment(auto_requeue=auto_requeue, requeue_signal=requeue_signal)

    @property
    def auto_requeue(self) -> bool:
        return self.slurm_impl.auto_requeue

    @property
    def requeue_signal(self) -> Optional[signal.Signals]:
        return self.slurm_impl.requeue_signal

    @property
    @override
    def creates_processes_externally(self) -> bool:
        return self.slurm_impl.creates_processes_externally

    @property
    @override
    def main_address(self) -> str:
        return self.slurm_impl.main_address

    @property
    @override
    def main_port(self) -> int:
        return self.slurm_impl.main_port

    @staticmethod
    @override
    def detect() -> bool:
        """Returns ``True`` if the current process was launched on a SLURM cluster.

        It is possible to use the SLURM scheduler to request resources and then launch processes manually using a
        different environment. For this, the user can set the job name in SLURM to 'bash' or 'interactive' (srun --job-
        name=interactive). This will then avoid the detection of ``SLURMEnvironment`` and another environment can be
        detected automatically.

        """
        SLURMEnvironment._validate_srun_used()
        return _is_srun_used()

    @staticmethod
    def job_name() -> Optional[str]:
        return os.environ.get("SLURM_JOB_NAME")

    @staticmethod
    def job_id() -> Optional[int]:
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.environments.slurm import (
            SLURMEnvironment as EnterpriseSLURMEnvironment,
        )

        return EnterpriseSLURMEnvironment.job_id()

    @override
    def world_size(self) -> int:
        return self.slurm_impl.world_size()

    @override
    def set_world_size(self, size: int) -> None:
        return self.slurm_impl.set_world_size(size)

    @override
    def global_rank(self) -> int:
        return self.slurm_impl.global_rank()

    @override
    def set_global_rank(self, rank: int) -> None:
        return self.slurm_impl.set_global_rank(rank)

    @override
    def local_rank(self) -> int:
        return self.slurm_impl.local_rank()

    @override
    def node_rank(self) -> int:
        return self.slurm_impl.node_rank()

    @override
    def validate_settings(self, num_devices: int, num_nodes: int) -> None:
        return self.slurm_impl.validate_settings(num_devices, num_nodes)

    @staticmethod
    def resolve_root_node_address(nodes: str) -> str:
        """The node selection format in SLURM supports several formats.

        This function selects the first host name from

        - a space-separated list of host names, e.g., 'host0 host1 host3' yields 'host0' as the root
        - a comma-separated list of host names, e.g., 'host0,host1,host3' yields 'host0' as the root
        - the range notation with brackets, e.g., 'host[5-9]' yields 'host5' as the root

        """
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.environments.slurm import (
            SLURMEnvironment as EnterpriseSLURMEnvironment,
        )

        return EnterpriseSLURMEnvironment.resolve_root_node_address(nodes)

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
        """Checks for conflicting or incorrectly set variables set through `srun` and raises a useful error message.

        Right now, we only check for the most common user errors. See
        `the srun docs <https://slurm.schedmd.com/srun.html>`_
        for a complete list of supported srun variables.

        """
        _raise_enterprise_not_available()
        from pytorch_lightning_enterprise.plugins.environments.slurm import (
            SLURMEnvironment as EnterpriseSLURMEnvironment,
        )

        return EnterpriseSLURMEnvironment._validate_srun_variables()


def _is_srun_used() -> bool:
    return "SLURM_NTASKS" in os.environ and not _is_slurm_interactive_mode()


def _is_slurm_interactive_mode() -> bool:
    return SLURMEnvironment.job_name() in ("bash", "interactive")
