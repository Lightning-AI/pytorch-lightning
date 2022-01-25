import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from pytorch_lightning.utilities.imports import _HYDRA_AVAILABLE
from tests.helpers.runif import RunIf

if _HYDRA_AVAILABLE:
    from omegaconf import OmegaConf


# fixture to run in a clean temporary directory
@pytest.fixture()
def cleandir():
    """Run function in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdirname:
        old_dir = os.getcwd()  # get current working directory (cwd)
        os.chdir(tmpdirname)  # change cwd to the temp-directory
        yield tmpdirname  # yields control to the test to be run
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
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)
        return stdout, stderr
    except Exception as e:
        cmd = " ".join(cmd)
        sys.stderr.write(f"Error executing:\n{cmd}\n")
        raise e


# Script to run from command line
script = """
import os

import hydra
import torch
from torch import distributed as dist

from pytorch_lightning import Trainer
from pytorch_lightning.strategies import DDPStrategy
from tests.helpers.boring_model import BoringModel


class BoringModelGPU(BoringModel):
    def on_train_start(self) -> None:
        # make sure that the model is on GPU when training
        assert self.device == torch.device(f"cuda:{self.trainer.strategy.local_rank}")
        self.start_cuda_memory = torch.cuda.memory_allocated()


@hydra.main()
def task_fn(cfg):
    trainer = Trainer(gpus=cfg.gpus, strategy=cfg.strategy, fast_dev_run=True)
    model = BoringModelGPU()
    trainer.fit(model)

    # Need to do this in addition to Lightning shutting down the
    # distributed processes in order to run a multirun loop with hydra
    if dist.is_initialized():
        dist.destroy_process_group()

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


if __name__ == "__main__":
    task_fn()
"""


@RunIf(skip_windows=True, min_gpus=2)
@pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="Hydra not Available")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("gpus", [1, 2])
@pytest.mark.parametrize("subdir", [None, "dksa", ".hello"])
def test_ddp_with_hydra_runjob(gpus, subdir):
    # Save script locally
    with open("temp.py", "w") as fn:
        fn.write(script)

    # Run CLI
    cmd = [sys.executable, "temp.py", f"+gpus={gpus}", '+strategy="ddp"']
    if subdir is not None:
        cmd += [f"hydra.output_subdir={subdir}"]
    run_process(cmd)

    # Make sure config.yaml was created
    logs = sorted(Path.cwd().glob("**/config.yaml"))
    assert len(logs) == 1

    # Make sure subdir was set
    actual_subdir = ".hydra" if subdir is None else subdir
    assert logs[0].parent.name == actual_subdir

    # Make sure the parameter was set and used
    cfg = OmegaConf.load(logs[0])
    assert cfg.gpus == gpus

    # Make sure PL spawned a job that is logged by Hydra
    logs = sorted(Path.cwd().glob("**/train_ddp_process_1.log"))
    assert len(logs) == gpus - 1


@RunIf(skip_windows=True, min_gpus=2)
@pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="Hydra not Available")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("gpus", [1, 2])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_ddp_with_hydra_multirunjob(gpus, num_jobs):
    with open("temp.py", "w") as fn:
        fn.write(script)

    fake_param = "+foo="
    for i in range(num_jobs):
        fake_param += f"{i}"
        if i < num_jobs - 1:
            fake_param += ","
    run_process([sys.executable, "temp.py", f"+gpus={gpus}", '+strategy="ddp"', fake_param, "--multirun"])

    configs = sorted(Path.cwd().glob("**/config.yaml"))
    assert len(configs) == num_jobs

    for i, config in enumerate(configs):
        cfg = OmegaConf.load(config)
        assert cfg.gpus == gpus
        assert cfg.foo == i

    logs = sorted(Path.cwd().glob("**/train_ddp_process_1.log"))
    assert len(logs) == num_jobs * (gpus - 1)
