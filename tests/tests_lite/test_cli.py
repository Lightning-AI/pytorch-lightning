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

import pytest
from tests_lite.helpers.runif import RunIf

from lightning_lite.cli import main as cli_main


@mock.patch("lightning_lite.cli.torchrun")
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_cli_env_vars_defaults(*_):
    with mock.patch("sys.argv", ["cli.py", "script.py"]):
        cli_main()
    assert os.environ["LT_CLI_USED"] == "1"
    assert os.environ["LT_ACCELERATOR"] == "cpu"
    assert "LT_STRATEGY" not in os.environ
    assert os.environ["LT_DEVICES"] == "1"
    assert os.environ["LT_NUM_NODES"] == "1"
    assert os.environ["LT_PRECISION"] == "32"


@pytest.mark.parametrize("accelerator", ["cpu", "gpu", "cuda", pytest.param("mps", marks=RunIf(mps=True))])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.cli.torchrun")
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_accelerator(_, __, accelerator):
    with mock.patch("sys.argv", ["cli.py", "script.py", "--accelerator", accelerator]):
        cli_main()
    assert os.environ["LT_ACCELERATOR"] == accelerator


@pytest.mark.parametrize("strategy", ["dp", "ddp", "deepspeed"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.cli.torchrun")
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_strategy(_, __, strategy):
    with mock.patch("sys.argv", ["cli.py", "script.py", "--strategy", strategy]):
        cli_main()
    assert os.environ["LT_STRATEGY"] == strategy


@pytest.mark.parametrize("devices", ["1", "2", "0,", "1,0", "-1"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.cli.torchrun")
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_cli_env_vars_devices_cuda(_, __, devices):
    with mock.patch("sys.argv", ["cli.py", "script.py", "--accelerator", "cuda", "--devices", devices]):
        cli_main()
    assert os.environ["LT_DEVICES"] == devices


@pytest.mark.parametrize("num_nodes", ["1", "2", "3"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.cli.torchrun")
def test_cli_env_vars_num_nodes(_, num_nodes):
    with mock.patch("sys.argv", ["cli.py", "script.py", "--num-nodes", num_nodes]):
        cli_main()
    assert os.environ["LT_NUM_NODES"] == num_nodes


@pytest.mark.parametrize("precision", ["64", "32", "16", "bf16"])
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.cli.torchrun")
def test_cli_env_vars_precision(_, precision):
    with mock.patch("sys.argv", ["cli.py", "script.py", "--precision", precision]):
        cli_main()
    assert os.environ["LT_PRECISION"] == precision


@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
@mock.patch("lightning_lite.cli.torchrun")
def test_cli_torchrun_defaults(torchrun_mock):
    with mock.patch("sys.argv", ["cli.py", "script.py"]):
        cli_main()
    torchrun_mock.main.assert_called_with(
        [
            "--nproc_per_node",
            "1",
            "--nnodes",
            "1",
            "--node_rank",
            "0",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            "29400",
            "script.py",
        ]
    )


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
@mock.patch("lightning_lite.cli.torchrun")
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=5)
def test_cli_torchrun_num_processes_launched(_, torchrun_mock, devices, expected):
    with mock.patch("sys.argv", ["cli.py", "script.py", "--accelerator", "cuda", "--devices", devices]):
        cli_main()

    torchrun_mock.main.assert_called_with(
        [
            "--nproc_per_node",
            str(expected),
            "--nnodes",
            "1",
            "--node_rank",
            "0",
            "--master_addr",
            "127.0.0.1",
            "--master_port",
            "29400",
            "script.py",
        ]
    )
