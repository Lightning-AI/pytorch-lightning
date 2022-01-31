import os
from typing import List, Optional, Tuple

import __main__
import torch

from pytorch_lightning.utilities import _HYDRA_AVAILABLE

if _HYDRA_AVAILABLE:
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd
    from omegaconf import OmegaConf


def get_ddp_spawn_command_for_hydra(command: List[str], local_rank: str) -> Tuple[str, str]:
    """Modifies the DDP spawn command to support Hydra initiated processes.

    If Hydra is initialized:   1) Set `cwd` to the hydra working directory   2) Use the stored configuration in
    `hydra_cfg.output_subdir / config.yaml` to spawn a new child
    """

    cwd: Optional[str] = None
    if HydraConfig.initialized():
        orig_cwd = get_original_cwd()
        cwd = os.getcwd()
        os_cwd = f'"{cwd}"'  # this is needed to handle characters like `=` in the directory name

        hydra_cfg = HydraConfig.get()
        hydra_output = os.path.join(os.path.relpath(cwd, orig_cwd), hydra_cfg.output_subdir)

        if __main__.__spec__ is None:  # pragma: no-cover
            command_no_args = command[:2]
        else:
            # this fails for `python -m pdb -m a.b.c <args>`
            command_no_args = command[:3]

        command = command_no_args
        command += ["-cp", hydra_output, "-cn", "config.yaml"]
        command += [
            f"hydra.output_subdir={hydra_cfg.output_subdir}",
            f"hydra.run.dir={os_cwd}",
            f"hydra.job.name=train_ddp_process_{local_rank}",
        ]
    return command, cwd


def teardown_ddp_for_hydra_multirun():
    """Performs additional teardown steps for PL to allow for Hydra multirun jobs."""

    if HydraConfig.initialized():
        hydra_cfg = HydraConfig.get()

        # check if we are in multirun mode
        if not OmegaConf.is_missing(hydra_cfg.job, "num"):
            # shutdown any distributed process groups
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            # Remove PL environments so next multirun starts fresh
            envs = (
                "LOCAL_RANK",
                "NODE_RANK",
                "WORLD_SIZE",
                "MASTER_ADDR",
                "MASTER_PORT",
                "PL_GLOBAL_SEED",
                "PL_SEED_WORKERS",
            )

            for name in envs:
                os.environ.pop(name, None)
