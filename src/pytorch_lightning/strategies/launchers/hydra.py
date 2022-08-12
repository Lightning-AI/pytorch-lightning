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
import subprocess
import sys
from time import sleep
from typing import Any, Callable, List, Optional, Tuple

import __main__
import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.utilities import _HYDRA_AVAILABLE

if _HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import to_absolute_path


class _HydraSubprocessScriptLauncher(_SubprocessScriptLauncher):
    """Hydra Launcher to support Hydra commands."""

    def _get_complete_path(self, command: str) -> str:
        return to_absolute_path(command)

    def _get_launch_command(self, command: List[str], local_rank: int) -> Tuple[List[str], Optional[str]]:
        """Modifies the command to support Hydra initiated processes."""
        if not HydraConfig.initialized():
            return command, None

        # If Hydra is initialized:
        #   1) Set `cwd` to the hydra working directory
        #   2) Use the stored configuration in `hydra_cfg.output_subdir / config.yaml` to spawn a new child

        cwd = os.getcwd()
        os_cwd = f'"{cwd}"'  # this is needed to handle characters like `=` in the directory name

        hydra_cfg = HydraConfig.get()
        hydra_output = os.path.join(cwd, hydra_cfg.output_subdir)

        if __main__.__spec__ is None:  # pragma: no-cover
            command_no_args = command[:2]
        else:
            # this fails for `python -m pdb -m a.b.c <args>`
            command_no_args = command[:3]

        command = command_no_args

        # run the Hydra job using the current job configuration
        # - typically located in:
        #        RUN MODE: hydra.run.dir/.hydra/config.ayml
        #        MULTIRUN MODE: hydra.sweep.dir/hydra.sweep.subdir/.hydra/config.yaml
        command += ["-cp", hydra_output, "-cn", "config.yaml"]

        # hydra.output_subdir=.pl_ddp_hydra_{local_rank}
        #   Store process config in its own to avoid overwriting
        #   and allow the user to very that each spawned job uses
        #   the same configuration
        # hydra.run.dir={os_cwd}
        #   This makes sure to run this job, log, and store any outputs
        #   in the current experiment directory
        #
        # hydra.job.name=train_ddp_process_{local_rank}
        #   This defines the logging output file for the process
        command += [
            f"hydra.output_subdir=.pl_ddp_hydra_{local_rank}",
            f"hydra.run.dir={os_cwd}",
            f"hydra.job.name=train_ddp_process_{local_rank}",
        ]
        return command, cwd

    def launch(self, function: Callable, *args: Any, trainer: Optional["pl.Trainer"] = None, **kwargs: Any) -> Any:
        """Creates new processes, then calls the given function.

        Arguments:
            function: A callback function to execute after all processes have been created.
                It is up to the implementation of this function to synchronize the processes, e.g., with barriers.
            *args: Optional positional arguments to be passed to the given function.
            trainer: Optional reference to the :class:`~pytorch_lightning.trainer.trainer.Trainer`.
            **kwargs: Optional keyword arguments to be passed to the given function.
        """
        results = super().launch(function, *args, **kwargs)
        _teardown_ddp_for_hydra_multirun()
        return results


def _teardown_ddp_for_hydra_multirun():
    if HydraConfig.initialized():
        # shutdown any distributed process groups
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        envs = (
            "LOCAL_RANK",
            "NODE_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
        )
        for name in envs:
            os.environ.pop(name, None)
