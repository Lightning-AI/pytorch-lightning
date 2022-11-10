# Copyright The PyTorch Lightning team.
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
import os
from unittest import mock
from unittest.mock import Mock

import pytest
from tests_lite.helpers.runif import RunIf

from lightning_lite.cli import _run_model
from lightning_lite.utilities.imports import _IS_WINDOWS, _TORCH_GREATER_EQUAL_1_13

if not (_IS_WINDOWS and _TORCH_GREATER_EQUAL_1_13):
    import torch.distributed.run


def skip_windows_pt_1_13():
    # https://github.com/pytorch/pytorch/issues/85427
    return pytest.mark.skipif(
        condition=(_IS_WINDOWS and _TORCH_GREATER_EQUAL_1_13),
        reason="Torchelastic import bug in 1.13 affecting Windows",
    )


@pytest.fixture
def fake_script(tmp_path):
    script = tmp_path / "script.py"
    script.touch()
    return str(script)


@skip_windows_pt_1_13()
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_defaults(monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script])
    assert e.value.code == 0
    assert os.environ["LT_CLI_USED"] == "1"
    assert os.environ["LT_ACCELERATOR"] == "cpu"
    assert "LT_STRATEGY" not in os.environ
    assert os.environ["LT_DEVICES"] == "1"
    assert os.environ["LT_NUM_NODES"] == "1"
    assert os.environ["LT_PRECISION"] == "32"


@skip_windows_pt_1_13()
@pytest.mark.parametrize("accelerator", ["cpu", "gpu", "cuda", pytest.param("mps", marks=RunIf(mps=True))])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_accelerator(_, accelerator, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--accelerator", accelerator])
    assert e.value.code == 0
    assert os.environ["LT_ACCELERATOR"] == accelerator


@skip_windows_pt_1_13()
@pytest.mark.parametrize("strategy", ["dp", "ddp", "deepspeed"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_strategy(_, strategy, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--strategy", strategy])
    assert e.value.code == 0
    assert os.environ["LT_STRATEGY"] == strategy


@skip_windows_pt_1_13()
@pytest.mark.parametrize("devices", ["1", "2", "0,", "1,0", "-1"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_devices_cuda(_, devices, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--accelerator", "cuda", "--devices", devices])
    assert e.value.code == 0
    assert os.environ["LT_DEVICES"] == devices


@RunIf(mps=True)
@skip_windows_pt_1_13()
@pytest.mark.parametrize("accelerator", ["mps", "gpu"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_devices_mps(accelerator, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--accelerator", accelerator])
    assert e.value.code == 0
    assert os.environ["LT_DEVICES"] == "1"


@skip_windows_pt_1_13()
@pytest.mark.parametrize("num_nodes", ["1", "2", "3"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_num_nodes(num_nodes, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--num-nodes", num_nodes])
    assert e.value.code == 0
    assert os.environ["LT_NUM_NODES"] == num_nodes


@skip_windows_pt_1_13()
@pytest.mark.parametrize("precision", ["64", "32", "16", "bf16"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_precision(precision, monkeypatch, fake_script):
    monkeypatch.setattr(torch.distributed, "run", Mock())
    with pytest.raises(SystemExit) as e:
        _run_model.main([fake_script, "--precision", precision])
    assert e.value.code == 0
    assert os.environ["LT_PRECISION"] == precision


@skip_windows_pt_1_13()
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


@skip_windows_pt_1_13()
@pytest.mark.parametrize(
    "devices,expected",
    [
        ("1", 1),
        ("2", 2),
        ("0,", 1),
        ("1,0,2", 3),
        ("-1", 5),
    ],
)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=5)
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
