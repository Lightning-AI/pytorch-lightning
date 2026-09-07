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
import subprocess
from typing import Any, Callable, Optional

from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

import lightning.pytorch as pl
from lightning.fabric.plugins import ClusterEnvironment
from lightning.fabric.strategies.launchers.subprocess_script import (
    _basic_subprocess_cmd,
    _hydra_subprocess_cmd,
    _launch_process_observer,
)
from lightning.fabric.utilities.distributed import _set_num_threads_if_needed
from lightning.pytorch.strategies.launchers.launcher import _Launcher
from lightning.pytorch.trainer.connectors.signal_connector import _SIGNUM

log = logging.getLogger(__name__)
_HYDRA_AVAILABLE = RequirementCache("hydra-core")


class _SubprocessScriptLauncher(_Launcher):
    r"""A process launcher that invokes the current script as many times as desired in a single node.

    This launcher needs to be invoked on each node.
    In its default behavior, the main process in each node then spawns N-1 child processes via :func:`subprocess.Popen`,
    where N is the number of devices (e.g. GPU) per node. It is very similar to how :mod:`torch.distributed.run`
    launches processes.

    For example, if the script gets invoked with the command

    .. code-block:: bash

        python train.py --devices 4

    The launcher will create three additional subprocesses that get called like so:

    .. code-block:: bash

        LOCAL_RANK=1 python train.py --devices 4
        LOCAL_RANK=2 python train.py --devices 4
        LOCAL_RANK=3 python train.py --devices 4

    It is implied that the main process which launched the others has ``LOCAL_RANK=0``.
    Beside the local rank, the following other environment variables also get set, but unlike the local rank, these
    get determined by the cluster environment:

    1. `MASTER_ADDR`: The IP address of the main node.
    2. `MASTER_PORT`: The port number of the main node through which all processes communicate.
    3. `NODE_RANK`: The index of the node the current process is running on. Ranges from 0 to ``num_nodes - 1``.
    4. `WORLD_SIZE`: The total number of processes across all nodes, i.e., ``num_processes * num_nodes``.

    Arguments:
        cluster_environment: A cluster environment that provides access to world size, node rank, etc.
        num_processes: The number of processes to launch in the current node.
        num_nodes: The total number of nodes that participate in this process group.

    """

    def __init__(self, cluster_environment: ClusterEnvironment, num_processes: int, num_nodes: int) -> None:
        super().__init__()
        self.cluster_environment = cluster_environment
        self.num_processes = num_processes
        self.num_nodes = num_nodes
        self.procs: list[subprocess.Popen] = []  # launched child subprocesses, does not include the launcher

    @property
    @override
    def is_interactive_compatible(self) -> bool:
        return False

    @override
    def launch(self, function: Callable, *args: Any, trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        """Creates new processes, then calls the given function.

        Arguments:
            function: A callback function to execute after all processes have been created.
                It is up to the implementation of this function to synchronize the processes, e.g., with barriers.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~lightning.pytorch.trainer.trainer.Trainer`.
            **kwargs: Optional keyword arguments to be passed to the given function.

        """
        self.cluster_environment.validate_settings(num_devices=self.num_processes, num_nodes=self.num_nodes)
        if not self.cluster_environment.creates_processes_externally:
            self._call_children_scripts()
            _launch_process_observer(self.procs)

        _set_num_threads_if_needed(num_processes=self.num_processes)
        return function(*args, **kwargs)

    @override
    def kill(self, signum: _SIGNUM) -> None:
        for proc in self.procs:
            log.debug(f"Process {os.getpid()} is terminating {proc.pid} with {signum}")
            # this skips subprocesses already terminated
            proc.send_signal(signum)

    def _call_children_scripts(self) -> None:
        # bookkeeping of spawned processes
        self._check_can_spawn_children()
        self.procs = []  # reset in case it's called twice

        # DDP Environment variables
        os.environ["MASTER_ADDR"] = self.cluster_environment.main_address
        os.environ["MASTER_PORT"] = str(self.cluster_environment.main_port)

        # allow the user to pass the node rank
        os.environ["NODE_RANK"] = str(self.cluster_environment.node_rank())
        os.environ["LOCAL_RANK"] = str(self.cluster_environment.local_rank())
        os.environ["WORLD_SIZE"] = f"{self.num_processes * self.num_nodes}"

        for local_rank in range(1, self.num_processes):
            env_copy = os.environ.copy()
            env_copy["LOCAL_RANK"] = f"{local_rank}"

            # remove env var if global seed not set
            if os.environ.get("PL_GLOBAL_SEED") is None and "PL_GLOBAL_SEED" in env_copy:
                del env_copy["PL_GLOBAL_SEED"]

            hydra_in_use = False
            cwd: Optional[str] = None
            if _HYDRA_AVAILABLE:
                from hydra.core.hydra_config import HydraConfig

                hydra_in_use = HydraConfig.initialized()

            if hydra_in_use:
                command, cwd = _hydra_subprocess_cmd(local_rank)
            else:
                command = _basic_subprocess_cmd()

            new_process = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.procs.append(new_process)

    def _check_can_spawn_children(self) -> None:
        if len(self.procs) > 0:
            raise RuntimeError("The launcher can only create subprocesses once.")
        if self.cluster_environment.local_rank() != 0:
            raise RuntimeError(
                "Lightning attempted to launch new distributed processes with `local_rank > 0`. This should not happen."
                " Possible reasons: 1) LOCAL_RANK environment variable was incorrectly modified by the user,"
                " 2) `ClusterEnvironment.creates_processes_externally` incorrectly implemented."
            )
