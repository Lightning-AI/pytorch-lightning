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
import os
from io import StringIO
from unittest import mock
from unittest.mock import Mock

import pytest
import torch.distributed.run

from lightning.fabric.cli import _get_supported_strategies, _run_model
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_1_12
from tests_fabric.helpers.runif import RunIf


@pytest.fixture()
def fake_script(tmp_path):
    script = tmp_path / "script.py"
    script.touch()
    return str(script)


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_defaults(monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script])
    assert e.value.code == 0
    assert os.environ["LT_CLI_USED"] == "1"
    assert "LT_ACCELERATOR" not in os.environ
    assert "LT_STRATEGY" not in os.environ
    assert os.environ["LT_DEVICES"] == "1"
    assert os.environ["LT_NUM_NODES"] == "1"
    assert "LT_PRECISION" not in os.environ


@pytest.mark.parametrize("accelerator", ["cpu", "gpu", "cuda", pytest.param("mps", marks=RunIf(mps=True))])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_accelerator(_, accelerator, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--accelerator", accelerator])
    assert e.value.code == 0
    assert os.environ["LT_ACCELERATOR"] == accelerator


@pytest.mark.parametrize("strategy", _get_supported_strategies())
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_strategy(_, strategy, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--strategy", strategy])
    assert e.value.code == 0
    assert os.environ["LT_STRATEGY"] == strategy


def test_cli_get_supported_strategies():
    """Test to ensure that when new strategies get added, we must consider updating the list of supported ones in the
    CLI."""
    if _TORCH_GREATER_EQUAL_1_12 and torch.distributed.is_available():
        assert len(_get_supported_strategies()) == 7
        assert "fsdp" in _get_supported_strategies()
    else:
        assert len(_get_supported_strategies()) == 6
        assert "fsdp" not in _get_supported_strategies()


@pytest.mark.parametrize("strategy", ["ddp_spawn", "ddp_fork", "ddp_notebook", "deepspeed_stage_3_offload"])
def test_cli_env_vars_unsupported_strategy(strategy, fake_script):
    ioerr = StringIO()
    with pytest.raises(SystemExit) as e, contextlib.redirect_stderr(ioerr):
        _run_model.main([fake_script, "--strategy", strategy])
    assert e.value.code == 2
    assert f"Invalid value for '--strategy': '{strategy}'" in ioerr.getvalue()


@pytest.mark.parametrize("devices", ["1", "2", "0,", "1,0", "-1"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_devices_cuda(_, devices, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--accelerator", "cuda", "--devices", devices])
    assert e.value.code == 0
    assert os.environ["LT_DEVICES"] == devices


@RunIf(mps=True)
@pytest.mark.parametrize("accelerator", ["mps", "gpu"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_devices_mps(accelerator, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--accelerator", accelerator])
    assert e.value.code == 0
    assert os.environ["LT_DEVICES"] == "1"


@pytest.mark.parametrize("num_nodes", ["1", "2", "3"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_num_nodes(num_nodes, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--num-nodes", num_nodes])
    assert e.value.code == 0
    assert os.environ["LT_NUM_NODES"] == num_nodes


@pytest.mark.parametrize("precision", ["64-true", "64", "32-true", "32", "16-mixed", "bf16-mixed"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_precision(precision, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--precision", precision])
    assert e.value.code == 0
    assert os.environ["LT_PRECISION"] == precision


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_torchrun_defaults(monkeypatch, fake_script):
    torchrun_mock = Mock()
    monkeypatch.setattr(torch.distributed, "run", torchrun_mock)
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script])
    assert e.value.code == 0
    torchrun_mock.main.assert_called_with(
        [
            "--nproc_per_node=1",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=127.0.0.1",
            "--master_port=29400",
            fake_script,
        ]
    )


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
def test_cli_torchrun_num_processes_launched(_, devices, expected, monkeypatch, fake_script):
    torchrun_mock = Mock()
    monkeypatch.setattr(torch.distributed, "run", torchrun_mock)
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--accelerator", "cuda", "--devices", devices])
    assert e.value.code == 0
    torchrun_mock.main.assert_called_with(
        [
            f"--nproc_per_node={expected}",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=127.0.0.1",
            "--master_port=29400",
            fake_script,
        ]
    )
