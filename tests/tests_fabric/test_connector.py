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
# limitations under the License
import inspect
import os
from typing import Any, Dict
from unittest import mock

import pytest
import torch
import torch.distributed
from tests_fabric.helpers.runif import RunIf

import lightning.fabric
from lightning.fabric import Fabric
from lightning.fabric.accelerators import TPUAccelerator
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.accelerators.cpu import CPUAccelerator
from lightning.fabric.accelerators.cuda import CUDAAccelerator
from lightning.fabric.accelerators.mps import MPSAccelerator
from lightning.fabric.connector import _Connector
from lightning.fabric.plugins import DoublePrecision, MixedPrecision, Precision, TPUPrecision
from lightning.fabric.plugins.environments import (
    KubeflowEnvironment,
    LightningEnvironment,
    LSFEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from lightning.fabric.plugins.io import TorchCheckpointIO
from lightning.fabric.strategies import (
    DataParallelStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    SingleDeviceStrategy,
    SingleTPUStrategy,
    XLAStrategy,
)
from lightning.fabric.strategies.ddp import _DDP_FORK_ALIASES
from lightning.fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning.fabric.utilities.exceptions import MisconfigurationException


def test_accelerator_choice_cpu():
    connector = _Connector()
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)


@RunIf(tpu=True, standalone=True)
@pytest.mark.parametrize(
    ["accelerator", "devices"], [("tpu", None), ("tpu", 1), ("tpu", [1]), ("tpu", 8), ("auto", 1), ("auto", 8)]
)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_accelerator_choice_tpu(accelerator, devices):
    connector = _Connector(accelerator=accelerator, devices=devices)
    assert isinstance(connector.accelerator, TPUAccelerator)
    if devices is None or (isinstance(devices, int) and devices > 1):
        # accelerator=tpu, devices=None (default) maps to devices=auto (8) and then chooses XLAStrategy
        # This behavior may change in the future: https://github.com/Lightning-AI/lightning/issues/10606
        assert isinstance(connector.strategy, XLAStrategy)
    else:
        assert isinstance(connector.strategy, SingleTPUStrategy)


@RunIf(skip_windows=True, standalone=True)
def test_strategy_choice_ddp_on_cpu():
    """Test that selecting DDPStrategy on CPU works."""
    _test_strategy_choice_ddp_and_cpu(ddp_strategy_class=DDPStrategy)


def _test_strategy_choice_ddp_and_cpu(ddp_strategy_class):
    connector = _Connector(
        strategy=ddp_strategy_class(),
        accelerator="cpu",
        devices=2,
    )
    assert isinstance(connector.strategy, ddp_strategy_class)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert connector.strategy.num_processes == 2
    assert connector.strategy.parallel_devices == [torch.device("cpu")] * 2


@mock.patch.dict(
    os.environ,
    {
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=0)
def test_custom_cluster_environment_in_slurm_environment(_):
    """Test that we choose the custom cluster even when SLURM or TE flags are around."""

    class CustomCluster(LightningEnvironment):
        @property
        def main_address(self):
            return "asdf"

        @property
        def creates_processes_externally(self) -> bool:
            return True

    connector = _Connector(
        plugins=[CustomCluster()],
        accelerator="cpu",
        strategy="ddp",
        devices=2,
    )
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, CustomCluster)


@RunIf(mps=False)
@mock.patch.dict(
    os.environ,
    {
        "SLURM_NTASKS": "2",
        "SLURM_NTASKS_PER_NODE": "1",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=0)
def test_custom_accelerator(*_):
    class Accel(Accelerator):
        def setup_device(self, device: torch.device) -> None:
            pass

        def get_device_stats(self, device: torch.device) -> Dict[str, Any]:
            pass

        def teardown(self) -> None:
            pass

        @staticmethod
        def parse_devices(devices):
            return devices

        @staticmethod
        def get_parallel_devices(devices):
            return [torch.device("cpu")] * devices

        @staticmethod
        def auto_device_count() -> int:
            return 1

        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def name() -> str:
            return "custom_acc_name"

    class Prec(Precision):
        pass

    class Strat(SingleDeviceStrategy):
        pass

    strategy = Strat(device=torch.device("cpu"), accelerator=Accel(), precision=Prec())
    connector = _Connector(strategy=strategy, devices=2)
    assert isinstance(connector.accelerator, Accel)
    assert isinstance(connector.strategy, Strat)
    assert isinstance(connector.precision, Prec)
    assert connector.strategy is strategy

    class Strat(DDPStrategy):
        pass

    strategy = Strat(accelerator=Accel(), precision=Prec())
    connector = _Connector(strategy=strategy, devices=2)
    assert isinstance(connector.accelerator, Accel)
    assert isinstance(connector.strategy, Strat)
    assert isinstance(connector.precision, Prec)
    assert connector.strategy is strategy


@pytest.mark.parametrize(
    "env_vars,expected_environment",
    [
        (
            {
                "SLURM_NTASKS": "2",
                "SLURM_NTASKS_PER_NODE": "1",
                "SLURM_JOB_NAME": "SOME_NAME",
                "SLURM_NODEID": "0",
                "LOCAL_RANK": "0",
                "SLURM_PROCID": "0",
                "SLURM_LOCALID": "0",
            },
            SLURMEnvironment,
        ),
        (
            {
                "LSB_JOBID": "1",
                "LSB_DJOB_RANKFILE": "SOME_RANK_FILE",
                "JSM_NAMESPACE_LOCAL_RANK": "1",
                "JSM_NAMESPACE_SIZE": "20",
                "JSM_NAMESPACE_RANK": "1",
            },
            LSFEnvironment,
        ),
    ],
)
@mock.patch("lightning.fabric.plugins.environments.lsf.LSFEnvironment._read_hosts", return_value=["node0", "node1"])
@mock.patch("lightning.fabric.plugins.environments.lsf.LSFEnvironment._get_node_rank", return_value=0)
def test_fallback_from_ddp_spawn_to_ddp_on_cluster(_, __, env_vars, expected_environment):
    with mock.patch.dict(os.environ, env_vars, clear=True):
        trainer = _Connector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, expected_environment)


@RunIf(mps=False)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_interactive_incompatible_backend_error(_, monkeypatch):
    monkeypatch.setattr(lightning.fabric.connector, "_IS_INTERACTIVE", True)
    with pytest.raises(RuntimeError, match=r"strategy='ddp'\)`.*is not compatible"):
        _Connector(strategy="ddp", accelerator="gpu", devices=2)

    with pytest.raises(RuntimeError, match=r"strategy='ddp_spawn'\)`.*is not compatible"):
        _Connector(strategy="ddp_spawn", accelerator="gpu", devices=2)

    with pytest.raises(RuntimeError, match=r"strategy='ddp'\)`.*is not compatible"):
        # Edge case: _Connector maps dp to ddp if accelerator != gpu
        _Connector(strategy="dp")


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_interactive_compatible_dp_strategy_gpu(_, __, monkeypatch):
    monkeypatch.setattr(lightning.fabric.utilities.imports, "_IS_INTERACTIVE", True)
    connector = _Connector(strategy="dp", accelerator="gpu")
    assert connector.strategy.launcher is None


@RunIf(skip_windows=True)
def test_interactive_compatible_strategy_tpu(tpu_available, monkeypatch):
    monkeypatch.setattr(lightning.fabric.utilities.imports, "_IS_INTERACTIVE", True)
    connector = _Connector(accelerator="tpu")
    assert connector.strategy.launcher.is_interactive_compatible


@RunIf(skip_windows=True)
def test_interactive_compatible_strategy_ddp_fork(monkeypatch):
    monkeypatch.setattr(lightning.fabric.utilities.imports, "_IS_INTERACTIVE", True)
    connector = _Connector(strategy="ddp_fork", accelerator="cpu")
    assert connector.strategy.launcher.is_interactive_compatible


@RunIf(mps=True)
@pytest.mark.parametrize(
    ["strategy", "strategy_class"],
    (
        ("ddp", DDPStrategy),
        ("dp", DataParallelStrategy),
        pytest.param("deepspeed", DeepSpeedStrategy, marks=RunIf(deepspeed=True)),
    ),
)
@pytest.mark.parametrize("accelerator", ["mps", "auto", "gpu", None, MPSAccelerator()])
def test_invalid_ddp_strategy_with_mps(accelerator, strategy, strategy_class):
    with pytest.raises(ValueError, match="strategies from the DDP family are not supported"):
        _Connector(accelerator=accelerator, strategy=strategy)

    with pytest.raises(ValueError, match="strategies from the DDP family are not supported"):
        _Connector(accelerator="mps", strategy=strategy_class())


@RunIf(mps=False)
@pytest.mark.parametrize(
    ["strategy", "strategy_class"],
    [
        ("ddp", DDPStrategy),
        ("ddp_spawn", DDPStrategy),
        pytest.param("deepspeed", DeepSpeedStrategy, marks=RunIf(deepspeed=True)),
    ],
)
@pytest.mark.parametrize("devices", [1, 2])
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_strategy_choice_multi_node_gpu(_, strategy, strategy_class, devices):
    connector = _Connector(num_nodes=2, accelerator="gpu", strategy=strategy, devices=devices)
    assert isinstance(connector.strategy, strategy_class)


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=0)
def test_cuda_accelerator_can_not_run_on_system(_):
    connector = _Connector(accelerator="cpu")
    assert isinstance(connector.accelerator, CPUAccelerator)

    with pytest.raises(
        RuntimeError,
        match="CUDAAccelerator` can not run on your system since the accelerator is not available.",
    ):
        _Connector(accelerator="cuda", devices=1)


@pytest.mark.skipif(TPUAccelerator.is_available(), reason="test requires missing TPU")
@mock.patch("lightning.fabric.accelerators.tpu._XLA_AVAILABLE", True)
def test_tpu_accelerator_can_not_run_on_system():
    with pytest.raises(RuntimeError, match="TPUAccelerator` can not run on your system"):
        _Connector(accelerator="tpu", devices=8)


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@pytest.mark.parametrize("device_count", (["0"], [0, "1"], ["GPU"], [["0", "1"], [0, 1]], [False]))
def test_accelererator_invalid_type_devices(_, device_count):
    with pytest.raises(
        MisconfigurationException, match=r"must be an int, a string, a sequence of ints or None, but you"
    ):
        _ = _Connector(accelerator="gpu", devices=device_count)


@RunIf(min_cuda_gpus=1)
def test_accelerator_gpu():
    connector = _Connector(accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)

    connector = _Connector(accelerator="gpu")
    assert isinstance(connector.accelerator, CUDAAccelerator)

    connector = _Connector(accelerator="auto", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)


@pytest.mark.parametrize(["devices", "strategy_class"], [(1, SingleDeviceStrategy), (5, DDPStrategy)])
def test_accelerator_cpu_with_devices(devices, strategy_class):
    connector = _Connector(accelerator="cpu", devices=devices)
    assert connector._parallel_devices == [torch.device("cpu")] * devices
    assert isinstance(connector.strategy, strategy_class)
    assert isinstance(connector.accelerator, CPUAccelerator)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ["devices", "strategy_class"], [(1, SingleDeviceStrategy), ([1], SingleDeviceStrategy), (2, DDPStrategy)]
)
def test_accelerator_gpu_with_devices(devices, strategy_class):
    connector = _Connector(accelerator="gpu", devices=devices)
    assert len(connector._parallel_devices) == len(devices) if isinstance(devices, list) else devices
    assert isinstance(connector.strategy, strategy_class)
    assert isinstance(connector.accelerator, CUDAAccelerator)


@RunIf(min_cuda_gpus=1)
def test_accelerator_auto_with_devices_gpu():
    connector = _Connector(accelerator="auto", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert connector._parallel_devices == [torch.device("cuda", 0)]


def test_set_devices_if_none_cpu():
    connector = _Connector(accelerator="cpu", devices=3)
    assert connector._parallel_devices == [torch.device("cpu")] * 3


@RunIf(mps=False)
def test_unsupported_strategy_types_on_cpu_and_fallback():
    with pytest.warns(UserWarning, match="is not supported on CPUs, hence setting `strategy='ddp"):
        connector = _Connector(strategy="dp", devices=2)
    assert isinstance(connector.strategy, DDPStrategy)


def test_invalid_accelerator_choice():
    with pytest.raises(ValueError, match="You selected an invalid accelerator name: `accelerator='cocofruit'`"):
        _Connector(accelerator="cocofruit")


@pytest.mark.parametrize("invalid_strategy", ["cocofruit", object()])
def test_invalid_strategy_choice(invalid_strategy):
    with pytest.raises(ValueError, match="You selected an invalid strategy name:"):
        _Connector(strategy=invalid_strategy)


@pytest.mark.parametrize(
    ["strategy", "strategy_class"],
    [
        ("ddp_spawn", DDPStrategy),
        ("ddp", DDPStrategy),
    ],
)
def test_strategy_choice_cpu_str(strategy, strategy_class):
    connector = _Connector(strategy=strategy, accelerator="cpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ["strategy", "strategy_class"],
    [
        ("ddp_spawn", DDPStrategy),
        ("ddp", DDPStrategy),
        ("dp", DataParallelStrategy),
        pytest.param("deepspeed", DeepSpeedStrategy, marks=RunIf(deepspeed=True)),
    ],
)
def test_strategy_choice_gpu_str(strategy, strategy_class):
    connector = _Connector(strategy=strategy, accelerator="gpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)


def test_device_type_when_strategy_instance_cpu_passed():
    connector = _Connector(strategy=DDPStrategy(), accelerator="cpu", devices=2)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.accelerator, CPUAccelerator)


@RunIf(min_cuda_gpus=2)
def test_device_type_when_strategy_instance_gpu_passed():
    connector = _Connector(strategy=DDPStrategy(), accelerator="gpu", devices=2)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.accelerator, CUDAAccelerator)


@pytest.mark.parametrize("precision", [1, 12, "invalid"])
def test_validate_precision_type(precision):
    with pytest.raises(ValueError, match=f"Precision {repr(precision)} is invalid"):
        _Connector(precision=precision)


def test_multi_device_default_strategy():
    """The default strategy when multiple devices are selected is "ddp" with the subprocess launcher."""
    connector = _Connector(strategy=None, accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert connector.strategy._start_method == "popen"
    assert isinstance(connector.strategy.launcher, _SubprocessScriptLauncher)


def test_strategy_choice_ddp_spawn_cpu():
    connector = _Connector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy._start_method == "spawn"
    assert connector.strategy.launcher._start_method == "spawn"


@RunIf(skip_windows=True)
@mock.patch("lightning.fabric.connector._IS_INTERACTIVE", True)
def test_strategy_choice_ddp_fork_in_interactive():
    """Test that when accelerator and strategy are unspecified, the connector chooses DDP Fork in interactive
    environments by default."""
    connector = _Connector(devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy._start_method == "fork"
    assert connector.strategy.launcher._start_method == "fork"


@RunIf(skip_windows=True)
def test_strategy_choice_ddp_fork_cpu():
    connector = _Connector(strategy="ddp_fork", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy._start_method == "fork"
    assert connector.strategy.launcher._start_method == "fork"


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp(*_):
    connector = _Connector(strategy="ddp", accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_spawn(*_):
    connector = _Connector(strategy="ddp_spawn", accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@pytest.mark.parametrize("job_name,expected_env", [("some_name", SLURMEnvironment), ("bash", LightningEnvironment)])
@pytest.mark.parametrize("strategy", ["ddp", DDPStrategy])
def test_strategy_choice_ddp_slurm(_, strategy, job_name, expected_env):
    if not isinstance(strategy, str):
        strategy = strategy()

    with mock.patch.dict(
        os.environ,
        {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "SLURM_NTASKS": "2",
            "SLURM_NTASKS_PER_NODE": "1",
            "SLURM_JOB_NAME": job_name,
            "SLURM_NODEID": "0",
            "SLURM_PROCID": "1",
            "SLURM_LOCALID": "1",
        },
    ):
        connector = _Connector(strategy=strategy, accelerator="cuda", devices=2)
        assert isinstance(connector.accelerator, CUDAAccelerator)
        assert isinstance(connector.strategy, DDPStrategy)
        assert isinstance(connector.strategy.cluster_environment, expected_env)


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
        "TORCHELASTIC_RUN_ID": "1",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_te(*_):
    connector = _Connector(strategy="ddp", accelerator="gpu", devices=2)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, TorchElasticEnvironment)
    assert connector.strategy.cluster_environment.local_rank() == 1
    assert connector.strategy.local_rank == 1


@mock.patch.dict(
    os.environ,
    {
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
        "TORCHELASTIC_RUN_ID": "1",
    },
)
def test_strategy_choice_ddp_cpu_te():
    connector = _Connector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, TorchElasticEnvironment)
    assert connector.strategy.cluster_environment.local_rank() == 1
    assert connector.strategy.local_rank == 1


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0",
        "KUBERNETES_PORT": "tcp://127.0.0.1:443",
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
    },
)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=1)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_kubeflow(*_):
    connector = _Connector(strategy="ddp", accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, KubeflowEnvironment)
    assert connector.strategy.cluster_environment.local_rank() == 0
    assert connector.strategy.local_rank == 0


@mock.patch.dict(
    os.environ,
    {
        "KUBERNETES_PORT": "tcp://127.0.0.1:443",
        "MASTER_ADDR": "1.2.3.4",
        "MASTER_PORT": "500",
        "WORLD_SIZE": "20",
        "RANK": "1",
    },
)
def test_strategy_choice_ddp_cpu_kubeflow():
    connector = _Connector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, KubeflowEnvironment)
    assert connector.strategy.cluster_environment.local_rank() == 0
    assert connector.strategy.local_rank == 0


@mock.patch.dict(
    os.environ,
    {
        "SLURM_NTASKS": "2",
        "SLURM_NTASKS_PER_NODE": "1",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    },
)
@pytest.mark.parametrize("strategy", ["ddp", DDPStrategy()])
def test_strategy_choice_ddp_cpu_slurm(strategy):
    connector = _Connector(strategy=strategy, accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, SLURMEnvironment)
    assert connector.strategy.local_rank == 0


@mock.patch.dict(os.environ, {}, clear=True)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_unsupported_tpu_choice(_, tpu_available):
    with pytest.raises(NotImplementedError, match=r"accelerator='tpu', precision=64\)` is not implemented"):
        _Connector(accelerator="tpu", precision=64)

    # if user didn't set strategy, _Connector will choose the TPUSingleStrategy or XLAStrategy
    with pytest.raises(ValueError, match="TPUAccelerator` can only be used with a `SingleTPUStrategy`"), pytest.warns(
        UserWarning, match=r"accelerator='tpu', precision=16\)` but AMP is not supported"
    ):
        _Connector(accelerator="tpu", precision=16, strategy="ddp")

    # wrong precision plugin type
    strategy = XLAStrategy(accelerator=TPUAccelerator(), precision=Precision())
    with pytest.raises(ValueError, match="TPUAccelerator` can only be used with a `TPUPrecision` plugin"):
        _Connector(strategy=strategy, devices=8)

    # wrong strategy type
    strategy = DDPStrategy(accelerator=TPUAccelerator(), precision=TPUPrecision())
    with pytest.raises(ValueError, match="TPUAccelerator` can only be used with a `SingleTPUStrategy`"):
        _Connector(strategy=strategy, devices=8)


@mock.patch("lightning.fabric.accelerators.cuda.CUDAAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_devices_auto_choice_cpu(tpu_available, *_):
    connector = _Connector(accelerator="auto", devices="auto")
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector.strategy.root_device == torch.device("cpu")


@RunIf(mps=False)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=2)
def test_devices_auto_choice_gpu(*_):
    connector = _Connector(accelerator="auto", devices="auto")
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert len(connector._parallel_devices) == 2


@RunIf(mps=True)
def test_devices_auto_choice_mps():
    connector = _Connector(accelerator="auto", devices="auto")
    assert isinstance(connector.accelerator, MPSAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector.strategy.root_device == torch.device("mps", 0)
    assert connector._parallel_devices == [torch.device("mps", 0)]


@pytest.mark.parametrize(
    ["parallel_devices", "accelerator"],
    [([torch.device("cpu")], "cuda"), ([torch.device("cuda", i) for i in range(8)], "tpu")],
)
def test_parallel_devices_in_strategy_conflict_with_accelerator(parallel_devices, accelerator):
    with pytest.raises(ValueError, match=r"parallel_devices set through"):
        _Connector(strategy=DDPStrategy(parallel_devices=parallel_devices), accelerator=accelerator)


@pytest.mark.parametrize(
    ["plugins", "expected"],
    [
        ([LightningEnvironment(), SLURMEnvironment()], "ClusterEnvironment"),
        ([TorchCheckpointIO(), TorchCheckpointIO()], "CheckpointIO"),
        (
            [Precision(), DoublePrecision(), LightningEnvironment(), SLURMEnvironment()],
            "Precision, ClusterEnvironment",
        ),
    ],
)
def test_plugin_only_one_instance_for_one_type(plugins, expected):
    with pytest.raises(ValueError, match=f"Received multiple values for {expected}"):
        _Connector(plugins=plugins)


@pytest.mark.parametrize("accelerator", ("cpu", "cuda", "mps", "tpu"))
@pytest.mark.parametrize("devices", ("0", 0, []))
def test_passing_zero_and_empty_list_to_devices_flag(accelerator, devices):
    with pytest.raises(ValueError, match="value is not a valid input using"):
        _Connector(accelerator=accelerator, devices=devices)


@pytest.mark.parametrize(
    "expected_accelerator_flag,expected_accelerator_class",
    [
        pytest.param("cuda", CUDAAccelerator, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", MPSAccelerator, marks=RunIf(mps=True)),
    ],
)
def test_gpu_accelerator_backend_choice(expected_accelerator_flag, expected_accelerator_class):
    connector = _Connector(accelerator="gpu")
    assert connector._accelerator_flag == expected_accelerator_flag
    assert isinstance(connector.accelerator, expected_accelerator_class)


@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=1)
def test_gpu_accelerator_backend_choice_cuda(*_):
    connector = _Connector(accelerator="gpu")
    assert connector._accelerator_flag == "cuda"
    assert isinstance(connector.accelerator, CUDAAccelerator)


@RunIf(min_torch="1.12")
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=True)
@mock.patch("lightning.fabric.accelerators.mps._get_all_available_mps_gpus", return_value=[0])
def test_gpu_accelerator_backend_choice_mps(*_):
    connector = _Connector(accelerator="gpu")
    assert connector._accelerator_flag == "mps"
    assert isinstance(connector.accelerator, MPSAccelerator)


@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning.fabric.accelerators.cuda.CUDAAccelerator.is_available", return_value=False)
def test_gpu_accelerator_no_gpu_backend_found_error(*_):
    with pytest.raises(RuntimeError, match="No supported gpu backend found!"):
        _Connector(accelerator="gpu")


@pytest.mark.parametrize("strategy", _DDP_FORK_ALIASES)
@mock.patch(
    "lightning.fabric.connector.torch.multiprocessing.get_all_start_methods",
    return_value=[],
)
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_ddp_fork_on_unsupported_platform(_, __, strategy):
    with pytest.raises(ValueError, match="process forking is not supported on this platform"):
        _Connector(strategy=strategy)


def test_precision_selection_16_on_cpu_warns():
    with pytest.warns(UserWarning, match=r"precision=16\)` but AMP is not supported on CPU. Using `precision='bf16"):
        _Connector(precision=16)


class MyAMP(MixedPrecision):
    pass


@RunIf(mps=False)
@pytest.mark.parametrize("strategy,devices", [("ddp", 2), ("ddp_spawn", 2)])
@pytest.mark.parametrize(
    "is_custom_plugin,plugin_cls",
    [(False, MixedPrecision), (True, MyAMP)],
)
def test_precision_selection_amp_ddp(strategy, devices, is_custom_plugin, plugin_cls):
    plugin = None
    if is_custom_plugin:
        plugin = plugin_cls(16, "cpu")
    connector = _Connector(
        precision=16,
        devices=devices,
        strategy=strategy,
        plugins=plugin,
    )
    assert isinstance(connector.precision, plugin_cls)


@pytest.mark.parametrize(["strategy", "strategy_cls"], [("DDP", DDPStrategy), ("Ddp", DDPStrategy)])
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_str_passed_being_case_insensitive(_, strategy, strategy_cls):
    connector = _Connector(strategy=strategy)
    assert isinstance(connector.strategy, strategy_cls)


@pytest.mark.parametrize("precision", ["64", "32", "16", "bf16"])
@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=1)
def test_precision_from_environment(_, precision):
    """Test that the precision input can be set through the environment variable."""
    with mock.patch.dict(os.environ, {"LT_PRECISION": precision}):
        connector = _Connector(accelerator="cuda")  # need to use cuda, because AMP not available on CPU
    assert isinstance(connector.precision, Precision)


@pytest.mark.parametrize(
    "accelerator, strategy, expected_accelerator, expected_strategy",
    [
        ("cpu", None, CPUAccelerator, SingleDeviceStrategy),
        ("cpu", "ddp", CPUAccelerator, DDPStrategy),
        pytest.param("mps", None, MPSAccelerator, SingleDeviceStrategy, marks=RunIf(mps=True)),
        pytest.param("cuda", "dp", CUDAAccelerator, DataParallelStrategy, marks=RunIf(min_cuda_gpus=1)),
        pytest.param(
            "cuda", "deepspeed", CUDAAccelerator, DeepSpeedStrategy, marks=RunIf(min_cuda_gpus=1, deepspeed=True)
        ),
    ],
)
def test_accelerator_strategy_from_environment(accelerator, strategy, expected_accelerator, expected_strategy):
    """Test that the accelerator and strategy input can be set through the environment variables."""
    env_vars = {"LT_ACCELERATOR": accelerator}
    if strategy is not None:
        env_vars["LT_STRATEGY"] = strategy

    with mock.patch.dict(os.environ, env_vars):
        connector = _Connector()
        assert isinstance(connector.accelerator, expected_accelerator)
        assert isinstance(connector.strategy, expected_strategy)


@mock.patch("lightning.fabric.accelerators.cuda.num_cuda_devices", return_value=8)
def test_devices_from_environment(*_):
    """Test that the devices and number of nodes can be set through the environment variables."""
    with mock.patch.dict(os.environ, {"LT_DEVICES": "2", "LT_NUM_NODES": "3"}):
        connector = _Connector(accelerator="cuda")
        assert isinstance(connector.accelerator, CUDAAccelerator)
        assert isinstance(connector.strategy, DDPStrategy)
        assert len(connector._parallel_devices) == 2
        assert connector._num_nodes_flag == 3


def test_arguments_from_environment_collision():
    """Test that the connector raises an error when the CLI settings conflict with settings in the code."""
    with mock.patch.dict(os.environ, {"LT_ACCELERATOR": "cpu"}):
        with pytest.raises(ValueError, match="`Fabric\\(accelerator='cuda', ...\\)` but .* `--accelerator=cpu`"):
            _Connector(accelerator="cuda")

    with mock.patch.dict(os.environ, {"LT_STRATEGY": "ddp"}):
        with pytest.raises(ValueError, match="`Fabric\\(strategy='ddp_spawn', ...\\)` but .* `--strategy=ddp`"):
            _Connector(strategy="ddp_spawn")

    with mock.patch.dict(os.environ, {"LT_DEVICES": "2"}):
        with pytest.raises(ValueError, match="`Fabric\\(devices=3, ...\\)` but .* `--devices=2`"):
            _Connector(devices=3)

    with mock.patch.dict(os.environ, {"LT_NUM_NODES": "3"}):
        with pytest.raises(ValueError, match="`Fabric\\(num_nodes=2, ...\\)` but .* `--num_nodes=3`"):
            _Connector(num_nodes=2)

    with mock.patch.dict(os.environ, {"LT_PRECISION": "16"}):
        with pytest.raises(ValueError, match="`Fabric\\(precision=64, ...\\)` but .* `--precision=16`"):
            _Connector(precision=64)


@RunIf(min_torch="1.12")
@mock.patch("lightning.fabric.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_fsdp_unsupported_on_cpu(_):
    """Test that we raise an error if attempting to run FSDP without GPU."""
    with pytest.raises(ValueError, match="You selected the FSDP strategy but FSDP is only available on GPU"):
        _Connector(strategy="fsdp")


def test_connector_defaults_match_fabric_defaults():
    """Test that the default values for the init arguments of Connector match the ones in Fabric."""

    def get_defaults(cls):
        init_signature = inspect.signature(cls)
        return {k: v.default for k, v in init_signature.parameters.items()}

    fabric_defaults = get_defaults(Fabric)
    connector_defaults = get_defaults(_Connector)

    # defaults should match on the intersection of argument names
    for name, connector_default in connector_defaults.items():
        assert connector_default == fabric_defaults[name]
