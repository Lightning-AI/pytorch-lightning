import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.imports import _HYDRA_AVAILABLE
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf

if _HYDRA_AVAILABLE:
    from hydra._internal.callbacks import Callbacks
    from hydra._internal.hydra import Hydra
    from hydra._internal.utils import create_config_search_path
    from hydra.types import HydraContext, RunMode
    from hydra.utils import instantiate
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
import torch

from pytorch_lightning import Trainer
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
    logs = list(Path.cwd().glob("**/config.yaml"))
    assert len(logs) == 1

    # Make sure subdir was set
    actual_subdir = ".hydra" if subdir is None else subdir
    assert logs[0].parent.name == actual_subdir

    # Make sure the parameter was set and used
    cfg = OmegaConf.load(logs[0])
    assert cfg.gpus == gpus

    # Make sure PL spawned a job that is logged by Hydra
    logs = list(Path.cwd().glob("**/train_ddp_process_1.log"))
    assert len(logs) == gpus - 1


@RunIf(skip_windows=True, min_gpus=2)
@pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="Hydra not Available")
@pytest.mark.usefixtures("cleandir")
@pytest.mark.parametrize("gpus", [1, 2])
@pytest.mark.parametrize("num_jobs", [1, 2])
def test_ddp_with_hydra_multirunjob(gpus, num_jobs):
    # Save script locally
    with open("temp.py", "w") as fn:
        fn.write(script)

    # create fake multirun params based on `num_jobs`
    fake_param = "+foo="
    for i in range(num_jobs):
        fake_param += f"{i}"
        if i < num_jobs - 1:
            fake_param += ","

    # Run CLI
    run_process([sys.executable, "temp.py", f"+gpus={gpus}", '+strategy="ddp"', fake_param, "--multirun"])

    # Make sure config.yaml was created for each job
    configs = sorted(Path.cwd().glob("**/config.yaml"))
    assert len(configs) == num_jobs

    # Make sure the parameter was set and used for each job
    for i, config in enumerate(configs):
        cfg = OmegaConf.load(config)
        assert cfg.gpus == gpus
        assert cfg.foo == i

    logs = list(Path.cwd().glob("**/train_ddp_process_1.log"))
    assert len(logs) == num_jobs * (gpus - 1)


def task_fn(cfg):
    trainer = Trainer(gpus=1, strategy="ddp", fast_dev_run=True)
    model = BoringModel()
    trainer.fit(model)


def run_hydra_sweeper():
    """Runs Hydra sweeper as a function (rather than CLI) so we can test teardown."""
    search_path = create_config_search_path(None)
    hydra = Hydra.create_main_hydra2(task_name="pytest", config_search_path=search_path)

    cfg = hydra.compose_config(
        config_name=None,
        overrides=[],
        with_log_configuration=False,
        run_mode=RunMode.MULTIRUN,
    )

    callbacks = Callbacks(cfg)
    # Instantiate sweeper without using Hydra's Plugin discovery (Zen!)
    sweeper = instantiate(cfg.hydra.sweeper)
    sweeper.setup(
        config=cfg,
        hydra_context=HydraContext(config_loader=hydra.config_loader, callbacks=callbacks),
        task_function=task_fn,
    )

    return sweeper.sweep(arguments=[])


@RunIf(skip_windows=True, min_gpus=2)
@pytest.mark.skipif(not _HYDRA_AVAILABLE, reason="Hydra not Available")
@pytest.mark.usefixtures("cleandir")
def test_ddp_teardown_with_hydra():
    job = run_hydra_sweeper()
    assert len(job) == 1

    # Make sure DDPPlugin.teardown executed
    #  - torch.distributed should be shutdown
    #  - PL environment variables are removed
    assert not torch.distributed.is_initialized()

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
        assert name not in os.environ
