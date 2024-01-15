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
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple

from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.plugins.environments.cluster_environment import ClusterEnvironment
from lightning.fabric.strategies.launchers.launcher import _Launcher
from lightning.fabric.utilities.distributed import _set_num_threads_if_needed
from lightning.fabric.utilities.rank_zero import rank_prefixed_message

_logger = logging.getLogger(__name__)
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

    def __init__(
        self,
        cluster_environment: "ClusterEnvironment",
        num_processes: int,
        num_nodes: int,
    ) -> None:
        super().__init__()
        self.cluster_environment = cluster_environment
        self.num_processes = num_processes
        self.num_nodes = num_nodes
        self.procs: List[subprocess.Popen] = []  # launched child subprocesses, does not include the launcher

    @property
    @override
    def is_interactive_compatible(self) -> bool:
        return False

    @override
    def launch(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        """Creates new processes, then calls the given function.

        Arguments:
            function: A callback function to execute after all processes have been created.
                It is up to the implementation of this function to synchronize the processes, e.g., with barriers.
            *args: Optional positional arguments to be passed to the given function.
            **kwargs: Optional keyword arguments to be passed to the given function.

        """
        self.cluster_environment.validate_settings(num_devices=self.num_processes, num_nodes=self.num_nodes)
        if not self.cluster_environment.creates_processes_externally:
            self._call_children_scripts()
            _launch_process_observer(self.procs)

        _set_num_threads_if_needed(num_processes=self.num_processes)
        return function(*args, **kwargs)

    def _call_children_scripts(self) -> None:
        # bookkeeping of spawned processes
        self._check_can_spawn_children()

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

            # start process
            # if hydra is available and initialized, make sure to set the cwd correctly
            hydra_in_use = False
            cwd: Optional[str] = None
            if _HYDRA_AVAILABLE:
                from hydra.core.hydra_config import HydraConfig

                hydra_in_use = HydraConfig.initialized()
            if hydra_in_use:
                command, cwd = _hydra_subprocess_cmd(local_rank=local_rank)
            else:
                command = _basic_subprocess_cmd()

            proc = subprocess.Popen(command, env=env_copy, cwd=cwd)
            self.procs.append(proc)

    def _check_can_spawn_children(self) -> None:
        if len(self.procs) > 0:
            raise RuntimeError("The launcher can only create subprocesses once.")
        if self.cluster_environment.local_rank() != 0:
            raise RuntimeError(
                "Lightning attempted to launch new distributed processes with `local_rank > 0`. This should not happen."
                " Possible reasons: 1) LOCAL_RANK environment variable was incorrectly modified by the user,"
                " 2) `ClusterEnvironment.creates_processes_externally` incorrectly implemented."
            )


def _basic_subprocess_cmd() -> Sequence[str]:
    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    if __main__.__spec__ is None:  # pragma: no-cover
        return [sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:]
    return [sys.executable, "-m", __main__.__spec__.name] + sys.argv[1:]


def _hydra_subprocess_cmd(local_rank: int) -> Tuple[Sequence[str], str]:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd, to_absolute_path

    import __main__  # local import to avoid https://github.com/Lightning-AI/lightning/issues/15218

    # when user is using hydra find the absolute path
    if __main__.__spec__ is None:  # pragma: no-cover
        command = [sys.executable, to_absolute_path(sys.argv[0])]
    else:
        command = [sys.executable, "-m", __main__.__spec__.name]

    command += sys.argv[1:]

    cwd = get_original_cwd()
    rundir = f'"{HydraConfig.get().run.dir}"'
    # Set output_subdir null since we don't want different subprocesses trying to write to config.yaml
    command += [f"hydra.run.dir={rundir}", f"hydra.job.name=train_ddp_process_{local_rank}", "hydra.output_subdir=null"]
    return command, cwd


def _launch_process_observer(child_processes: List[subprocess.Popen]) -> None:
    """Launches a thread that runs along the main process and monitors the health of all processes."""
    _ChildProcessObserver(child_processes=child_processes, main_pid=os.getpid()).start()


class _ChildProcessObserver(threading.Thread):
    def __init__(self, main_pid: int, child_processes: List[subprocess.Popen], sleep_period: int = 5) -> None:
        super().__init__(daemon=True, name="child-process-observer")  # thread stops if the main process exits
        self._main_pid = main_pid
        self._child_processes = child_processes
        self._sleep_period = sleep_period
        # Note: SIGTERM is not aggressive enough to terminate processes hanging in collectives
        self._termination_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
        self._finished = False

    @override
    def run(self) -> None:
        while not self._finished:
            time.sleep(self._sleep_period)
            self._finished = self._run()

    def _run(self) -> bool:
        """Runs once over all child processes to check whether they are still running."""
        for proc in self._child_processes:
            proc.poll()

        return_codes = [proc.returncode for proc in self._child_processes]
        if all(return_code == 0 for return_code in return_codes):
            return True

        for i, proc in enumerate(self._child_processes):
            if proc.returncode:
                message = rank_prefixed_message(
                    f"Child process with PID {proc.pid} terminated with code {proc.returncode}."
                    f" Forcefully terminating all other processes to avoid zombies 🧟",
                    rank=(i + 1),
                )
                _logger.info(message)
                self._terminate_all()
                return True

        return False

    def _terminate_all(self) -> None:
        """Terminates the main process and all its children."""
        for p in self._child_processes:
            p.send_signal(self._termination_signal)
        os.kill(self._main_pid, self._termination_signal)
