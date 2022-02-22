import os
from typing import List, Optional, Tuple

import __main__
import torch

from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.utilities import _HYDRA_AVAILABLE

if _HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import to_absolute_path


class _SubprocessScriptHydraLauncher(_SubprocessScriptLauncher):
    """Hydra Launcher subclasses `_SubprocessScriptLauncher to manage Hydra specific commands when running under
    Hydra environment."""

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

    def teardown(self) -> None:
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
