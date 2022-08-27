import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from pytorch_lightning.strategies.launchers.subprocess_script import _HYDRA_AVAILABLE
from pytorch_lightning.utilities.imports import _RequirementAvailable
from tests_pytorch.helpers.runif import RunIf

_HYDRA_WITH_RERUN = _RequirementAvailable("hydra-core >= 1.2")

if _HYDRA_AVAILABLE:
    from omegaconf import OmegaConf


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


# function to run a command line argument
def run_process(cmd):
    try:
        process = subprocess.Popen(
            args=cmd,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            sys.stderr.write(f"Subprocess error:\n{stderr}\n")
            sys.stderr.write(f"Subprocess stdout:\n{stdout}\n")
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
        return stdout, stderr
    except Exception as e:
        cmd = " ".join(cmd)
        sys.stderr.write(f"Error executing:\n{cmd}\n")
        raise e


# Script to run from command line
script = """
import hydra
import os
import torch

from pytorch_lightning import Trainer

from tests.tests_pytorch.helpers import BoringModel

class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.strategy.local_rank}")
        self.start_cuda_memory = torch.cuda.memory_allocated()

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


@RunIf(min_cuda_gpus=2)
@pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="Hydra not Available")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("subdir", [None, "dksa", ".hello"])
def test_ddp_with_hydra_runjob(subdir):
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


@RunIf(min_cuda_gpus=2)
@pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="Hydra not Available")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_ddp_with_hydra_multirunjob(num_jobs):
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


@RunIf(min_cuda_gpus=2)
@pytest.mark.skipif(not _HYDRA_WITH_RERUN, reason="Hydra with `rerun` not Available")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_ddp_with_hydra_multirunjob_rerun(num_jobs):
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
