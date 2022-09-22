import logging
import os
import sys
from pathlib import Path

import pytest
from lightning_utilities.core.imports import RequirementCache

from pytorch_lightning.strategies.launchers.subprocess_script import _HYDRA_AVAILABLE
from tests_pytorch.helpers.runif import RunIf

_HYDRA_WITH_RERUN = RequirementCache("hydra-core>=1.2")
_HYDRA_WITH_RUN_PROCESS = RequirementCache("hydra-core>=1.0.7")

if _HYDRA_AVAILABLE:
    from omegaconf import OmegaConf
if _HYDRA_WITH_RUN_PROCESS:
    from hydra.test_utils.test_utils import run_process


# fixture to run hydra jobs in a clean temporary directory
# Hydra creates its own output directories and logs
@pytest.fixture
def cleandir(tmp_path):
    """Run function in a temporary directory."""
    old_dir = os.getcwd()  # get current working directory (cwd)
    os.chdir(tmp_path)  # change cwd to the temp-directory
    yield tmp_path  # yields control to the test to be run
    os.chdir(old_dir)
    logging.shutdown()


# Script to run from command line
script = """
import hydra
import os
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel

class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.strategy.local_rank}")

@hydra.main(config_path=None, version_base="1.1")
def task_fn(cfg):
    trainer = Trainer(accelerator="auto", devices=cfg.devices, strategy=cfg.strategy, fast_dev_run=True)
    model = BoringModelGPU()
    trainer.fit(model)
    trainer.test(model)

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

    os.environ.pop("LOCAL_RANK", None)

if __name__ == "__main__":
    task_fn()
"""


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.skipif(not _HYDRA_WITH_RUN_PROCESS, reason=str(_HYDRA_WITH_RUN_PROCESS))
@pytest.mark.parametrize("subdir", [None, "dksa", ".hello"])
def test_ddp_with_hydra_runjob(cleandir, subdir):
    # Save script locally
    with open("temp.py", "w") as fn:
        fn.write(script)

    # Run CLI
    devices = 2
    cmd = [sys.executable, "temp.py", f"+devices={devices}", '+strategy="ddp"']
    if subdir is not None:
        cmd += [f"hydra.output_subdir={subdir}"]
    run_process(cmd)

    # Make sure config.yaml was created for additional
    # processes.
    logs = list(Path.cwd().glob("**/config.yaml"))
    assert len(logs) == devices

    # Make sure the parameter was set and used
    cfg = OmegaConf.load(logs[0])
    assert cfg.devices == devices

    # Make sure PL spawned a job that is logged by Hydra
    logs = list(Path.cwd().glob("**/*.log"))
    assert len(logs) == 1


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.skipif(not _HYDRA_WITH_RUN_PROCESS, reason=str(_HYDRA_WITH_RUN_PROCESS))
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_ddp_with_hydra_multirunjob(cleandir, num_jobs):
    # Save script locally
    with open("temp.py", "w") as fn:
        fn.write(script)

    # create fake multirun params based on `num_jobs`
    fake_param = "+foo=" + ",".join(str(i) for i in range(num_jobs))

    # Run CLI
    run_process([sys.executable, "temp.py", "+devices=2", '+strategy="ddp"', fake_param, "--multirun"])

    # Make sure config.yaml was created for each job
    configs = sorted(Path.cwd().glob("**/.pl_ddp_hydra_*/config.yaml"))
    assert len(configs) == num_jobs

    # Make sure the parameter was set and used for each job
    for i, config in enumerate(configs):
        cfg = OmegaConf.load(config)
        local_rank = int(config.parent.parent.parts[-1])
        assert cfg.devices == 2
        assert cfg.foo == local_rank

    logs = list(Path.cwd().glob("**/*.log"))
    assert len(logs) == num_jobs


yaml_file = """
hydra:
  callbacks:
    save_job_info:
      _target_: hydra.experimental.callbacks.PickleJobInfoCallback
"""


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.skipif(not _HYDRA_WITH_RERUN, reason=str(_HYDRA_WITH_RERUN))
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_ddp_with_hydra_multirunjob_rerun(cleandir, num_jobs):
    # Save script locally
    with open("temp.py", "w") as fn:
        fn.write(script)

    with open("config.yaml", "w") as fn:
        fn.write(yaml_file)

    # create fake multirun params based on `num_jobs`
    fake_param = "+foo=" + ",".join(str(i) for i in range(num_jobs))

    # Run CLI
    run_process(
        [
            sys.executable,
            "temp.py",
            "-cp",
            ".",
            "-cn",
            "config.yaml",
            "+devices=2",
            '+strategy="ddp"',
            fake_param,
            "--multirun",
        ]
    )

    pickles = sorted(Path.cwd().glob("**/.hydra/config.pickle"))
    assert len(pickles) == num_jobs
