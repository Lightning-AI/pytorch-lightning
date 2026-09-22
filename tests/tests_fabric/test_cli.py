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
import contextlib
import json
import logging
import os
import socket
import subprocess
import sys
from io import StringIO
from pathlib import Path
from unittest import mock
from unittest.mock import Mock

import pytest

from lightning.fabric.cli import _consolidate, _get_supported_strategies, _run
from lightning.fabric.utilities.load import _METADATA_FILENAME
from tests_fabric.helpers.runif import RunIf


@pytest.fixture
def fake_script(tmp_path):
    script = tmp_path / "script.py"
    script.touch()
    return str(script)


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_run_env_vars_defaults(monkeypatch, fake_script):
    monkeypatch.setitem(sys.modules, "torch.distributed.run", Mock())
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script])
    assert e.value.code == 0
    assert os.environ["LT_CLI_USED"] == "1"
    assert "LT_ACCELERATOR" not in os.environ
    assert "LT_STRATEGY" not in os.environ
    assert os.environ["LT_DEVICES"] == "1"
    assert os.environ["LT_NUM_NODES"] == "1"
    assert "LT_PRECISION" not in os.environ


@pytest.mark.parametrize("accelerator", ["cpu", "gpu", "cuda", "auto", pytest.param("mps", marks=RunIf(mps=True))])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_run_env_vars_accelerator(_, accelerator, monkeypatch, fake_script):
    monkeypatch.setitem(sys.modules, "torch.distributed.run", Mock())
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script, "--accelerator", accelerator])
    assert e.value.code == 0
    assert os.environ["LT_ACCELERATOR"] == accelerator


@pytest.mark.parametrize("strategy", _get_supported_strategies())
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_run_env_vars_strategy(_, strategy, monkeypatch, fake_script):
    monkeypatch.setitem(sys.modules, "torch.distributed.run", Mock())
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script, "--strategy", strategy])
    assert e.value.code == 0
    assert os.environ["LT_STRATEGY"] == strategy


def test_run_get_supported_strategies():
    """Test to ensure that when new strategies get added, we must consider updating the list of supported ones in the
    CLI."""
    assert len(_get_supported_strategies()) == 8
    assert "fsdp" in _get_supported_strategies()
    assert "ddp_find_unused_parameters_true" in _get_supported_strategies()


@pytest.mark.parametrize("strategy", ["ddp_spawn", "ddp_fork", "ddp_notebook", "deepspeed_stage_3_offload"])
def test_run_env_vars_unsupported_strategy(strategy, fake_script):
    ioerr = StringIO()
    with pytest.raises(SystemExit) as e, contextlib.redirect_stderr(ioerr):
        _run.main([fake_script, "--strategy", strategy])
    assert e.value.code == 2
    assert f"Invalid value for '--strategy': '{strategy}'" in ioerr.getvalue()


@pytest.mark.parametrize("devices", ["1", "2", "0,", "1,0", "-1", "auto"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_run_env_vars_devices_cuda(_, devices, monkeypatch, fake_script):
    monkeypatch.setitem(sys.modules, "torch.distributed.run", Mock())
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script, "--accelerator", "cuda", "--devices", devices])
    assert e.value.code == 0
    assert os.environ["LT_DEVICES"] == devices


@RunIf(mps=True)
@pytest.mark.parametrize("accelerator", ["mps", "gpu", "auto"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_run_env_vars_devices_mps(accelerator, monkeypatch, fake_script):
    monkeypatch.setitem(sys.modules, "torch.distributed.run", Mock())
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script, "--accelerator", accelerator])
    assert e.value.code == 0
    assert os.environ["LT_DEVICES"] == "1"


@pytest.mark.parametrize("num_nodes", ["1", "2", "3"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_run_env_vars_num_nodes(num_nodes, monkeypatch, fake_script):
    monkeypatch.setitem(sys.modules, "torch.distributed.run", Mock())
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script, "--num-nodes", num_nodes])
    assert e.value.code == 0
    assert os.environ["LT_NUM_NODES"] == num_nodes


@pytest.mark.parametrize("precision", ["64-true", "64", "32-true", "32", "16-mixed", "bf16-mixed"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_run_env_vars_precision(precision, monkeypatch, fake_script):
    monkeypatch.setitem(sys.modules, "torch.distributed.run", Mock())
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script, "--precision", precision])
    assert e.value.code == 0
    assert os.environ["LT_PRECISION"] == precision


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_run_torchrun_defaults(monkeypatch, fake_script):
    torchrun_mock = Mock()
    monkeypatch.setitem(sys.modules, "torch.distributed.run", torchrun_mock)
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script])
    assert e.value.code == 0
    torchrun_mock.main.assert_called_with([
        "--nproc_per_node=1",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        "--master_port=29400",
        fake_script,
    ])


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        ("1", 1),
        ("2", 2),
        ("0,", 1),
        ("1,0,2", 3),
        ("-1", 5),
    ],
)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=5)
def test_run_torchrun_num_processes_launched(_, devices, expected, monkeypatch, fake_script):
    torchrun_mock = Mock()
    monkeypatch.setitem(sys.modules, "torch.distributed.run", torchrun_mock)
    with pytest.raises(SystemExit) as e:
        _run.main([fake_script, "--accelerator", "cuda", "--devices", devices])
    assert e.value.code == 0
    torchrun_mock.main.assert_called_with([
        f"--nproc_per_node={expected}",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        "--master_port=29400",
        fake_script,
    ])


def test_run_through_fabric_entry_point():
    result = subprocess.run("fabric run --help", capture_output=True, text=True, shell=True)

    message = "Usage: fabric run [OPTIONS] SCRIPT [SCRIPT_ARGS]"
    assert message in result.stdout or message in result.stderr


@mock.patch("lightning.fabric.cli._load_distributed_checkpoint")
@mock.patch("lightning.fabric.cli._atomic_save")
def test_consolidate(save_mock, _, tmp_path, caplog, monkeypatch):
    # The checkpoint folder is validated by `_process_cli_args`, not click, so that remote (fsspec) paths
    # that don't exist as local files are not rejected before the real (fsspec-aware) check runs.
    with (
        caplog.at_level(logging.ERROR, logger="lightning.fabric.utilities.consolidate_checkpoint"),
        pytest.raises(SystemExit) as e,
    ):
        _consolidate.main(["not exist"])
    assert e.value.code == 1
    assert "checkpoint folder does not exist" in caplog.text

    checkpoint_folder = tmp_path / "checkpoint"
    checkpoint_folder.mkdir()
    (checkpoint_folder / _METADATA_FILENAME).touch()
    ioerr = StringIO()
    with pytest.raises(SystemExit) as e, contextlib.redirect_stderr(ioerr):
        _consolidate.main([str(checkpoint_folder)])
    assert e.value.code == 0
    save_mock.assert_called_once()


@pytest.mark.parametrize("module_flag", ["-m", "--module"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_run_torchrun_module(module_flag, monkeypatch):
    torchrun_mock = Mock()
    monkeypatch.setitem(sys.modules, "torch.distributed.run", torchrun_mock)
    with pytest.raises(SystemExit) as ex:
        _run.main(["--accelerator=cpu", module_flag, "package.train", "--", "--module", "value with spaces"])
    assert ex.value.code == 0
    torchrun_mock.main.assert_called_once_with([
        "--nproc_per_node=1",
        "--nnodes=1",
        "--node_rank=0",
        "--master_addr=127.0.0.1",
        "--master_port=29400",
        "--module",
        "package.train",
        "--module",
        "value with spaces",
    ])


def test_run_missing_script(monkeypatch):
    torchrun_mock = Mock()
    monkeypatch.setitem(sys.modules, "torch.distributed.run", torchrun_mock)
    ioerr = StringIO()
    with pytest.raises(SystemExit) as ex, contextlib.redirect_stderr(ioerr):
        _run.main(["missing_script.py"])
    assert ex.value.code == 2
    assert "Invalid value for 'SCRIPT'" in ioerr.getvalue()
    assert "does not exist" in ioerr.getvalue()
    torchrun_mock.main.assert_not_called()


@pytest.mark.parametrize("module_name", ["training_package.train", "training_package"])
def test_run_module_distributed(tmp_path, module_name):
    package = tmp_path / "training_package"
    package.mkdir()
    (package / "__init__.py").touch()
    (package / "constants.py").write_text("VALUE = 'relative import succeeded'\n")
    script = """
import json
import sys
from pathlib import Path

import torch
from lightning.fabric import Fabric

from .constants import VALUE

fabric = Fabric()
value = fabric.all_reduce(torch.tensor(float(fabric.global_rank + 1)), reduce_op="sum")
Path(f"rank_{fabric.global_rank}.json").write_text(json.dumps({
    "value": value.item(),
    "world_size": fabric.world_size,
    "relative_import": VALUE,
    "args": sys.argv[1:],
}))
"""
    (package / "train.py").write_text(script)
    (package / "__main__.py").write_text(script)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    # Keep the source checkout importable after changing to the temporary package directory.
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, env.get("PYTHONPATH")]))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "lightning.fabric.cli",
            "--accelerator=cpu",
            "--devices=2",
            f"--main-port={port}",
            "--module",
            module_name,
            "--",
            "--message",
            "value with spaces",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for rank in range(2):
        assert json.loads((tmp_path / f"rank_{rank}.json").read_text()) == {
            "value": 3.0,
            "world_size": 2,
            "relative_import": "relative import succeeded",
            "args": ["--message", "value with spaces"],
        }
