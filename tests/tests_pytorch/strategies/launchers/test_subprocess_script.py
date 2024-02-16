import subprocess
import sys
from unittest import mock
from unittest.mock import Mock

import pytest
from lightning.pytorch.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_utilities.core.imports import RequirementCache

from tests_pytorch.helpers.runif import RunIf

_HYDRA_WITH_RUN_PROCESS = RequirementCache("hydra-core>=1.0.7")

if _HYDRA_WITH_RUN_PROCESS:
    from hydra.test_utils.test_utils import run_process
    from omegaconf import OmegaConf


# Script to run from command line
script = """
import hydra
import os
import torch

from lightning.fabric.utilities.distributed import _distributed_is_initialized
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

    if _distributed_is_initialized():
        torch.distributed.destroy_process_group()

    os.environ.pop("LOCAL_RANK", None)

if __name__ == "__main__":
    task_fn()
"""


@RunIf(min_cuda_gpus=2, skip_windows=True, standalone=True)
@pytest.mark.skipif(not _HYDRA_WITH_RUN_PROCESS, reason=str(_HYDRA_WITH_RUN_PROCESS))
@pytest.mark.parametrize("subdir", [None, "null", "dksa", ".hello"])
def test_ddp_with_hydra_runjob(subdir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Save script locally
    with open("temp.py", "w") as fn:
        fn.write(script)

    # Run CLI
    devices = 2
    run_dir = tmp_path / "hydra_output"
    cmd = [sys.executable, "temp.py", f"+devices={devices}", '+strategy="ddp"', f"hydra.run.dir={run_dir}"]
    if subdir is not None:
        cmd += [f"hydra.output_subdir={subdir}"]
    run_process(cmd)

    # Make sure no config.yaml was created for additional processes
    saved_confs = list(run_dir.glob("**/config.yaml"))
    assert len(saved_confs) == (0 if subdir == "null" else 1)  # Main process has config.yaml iff subdir!="null"

    if saved_confs:  # Make sure the parameter was set and used
        cfg = OmegaConf.load(saved_confs[0])
        assert cfg.devices == devices

    # Make sure PL spawned jobs that are logged by Hydra
    logs = list(run_dir.glob("**/*.log"))
    assert len(logs) == devices


@mock.patch("lightning.fabric.strategies.launchers.subprocess_script._ChildProcessObserver")
def test_kill(_):
    launcher = _SubprocessScriptLauncher(Mock(), 1, 1)
    proc0 = Mock(autospec=subprocess.Popen)
    proc1 = Mock(autospec=subprocess.Popen)
    launcher.procs = [proc0, proc1]

    launcher.kill(15)
    proc0.send_signal.assert_called_once_with(15)
    proc1.send_signal.assert_called_once_with(15)


@mock.patch("lightning.fabric.strategies.launchers.subprocess_script.subprocess.Popen")
@mock.patch("lightning.fabric.strategies.launchers.subprocess_script._ChildProcessObserver")
def test_validate_cluster_environment_user_settings(*_):
    """Test that the launcher calls into the cluster environment to validate the user settings."""
    cluster_env = Mock(validate_settings=Mock(side_effect=RuntimeError("test")))
    cluster_env.creates_processes_externally = True
    launcher = _SubprocessScriptLauncher(cluster_env, num_processes=2, num_nodes=1)

    with pytest.raises(RuntimeError, match="test"):
        launcher.launch(Mock())
