import subprocess
import sys
from unittest.mock import Mock

import pytest
from lightning_utilities.core.imports import RequirementCache

from lightning.pytorch.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from tests_pytorch.helpers.runif import RunIf

_HYDRA_WITH_RERUN = RequirementCache("hydra-core>=1.2")
_HYDRA_WITH_RUN_PROCESS = RequirementCache("hydra-core>=1.0.7")

if _HYDRA_WITH_RUN_PROCESS:
    from hydra.test_utils.test_utils import run_process


# Script to run from command line
script = """
import hydra
import os
import torch

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel

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
def test_ddp_with_hydra_runjob(subdir, tmpdir, monkeypatch):
    monkeypatch.chdir(tmpdir)

    # Save script locally
    with open("temp.py", "w") as fn:
        fn.write(script)

    # Run CLI
    devices = 2
    cmd = [sys.executable, "temp.py", f"+devices={devices}", '+strategy="ddp"']
    if subdir is not None:
        cmd += [f"hydra.output_subdir={subdir}"]
    run_process(cmd)


def test_kill():
    launcher = _SubprocessScriptLauncher(Mock(), 1, 1)
    proc0 = Mock(autospec=subprocess.Popen)
    proc1 = Mock(autospec=subprocess.Popen)
    launcher.procs = [proc0, proc1]

    launcher.kill(15)
    proc0.send_signal.assert_called_once_with(15)
    proc1.send_signal.assert_called_once_with(15)
