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
# limitations under the License

import os
from typing import Any, Dict
from unittest import mock

import pytest
import torch
import torch.distributed
from tests_lite.helpers.runif import RunIf

import lightning_lite
from lightning_lite.accelerators import TPUAccelerator
from lightning_lite.accelerators.accelerator import Accelerator
from lightning_lite.accelerators.cpu import CPUAccelerator
from lightning_lite.accelerators.cuda import CUDAAccelerator
from lightning_lite.accelerators.mps import MPSAccelerator
from lightning_lite.connector import _Connector
from lightning_lite.plugins import DoublePrecision, NativeMixedPrecision, Precision, TPUPrecision
from lightning_lite.plugins.environments import (
    KubeflowEnvironment,
    LightningEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from lightning_lite.plugins.io import TorchCheckpointIO
from lightning_lite.strategies import (
    DataParallelStrategy,
    DDPShardedStrategy,
    DDPSpawnShardedStrategy,
    DDPSpawnStrategy,
    DDPStrategy,
    DeepSpeedStrategy,
    SingleDeviceStrategy,
    SingleTPUStrategy,
    XLAStrategy,
)
from lightning_lite.strategies.ddp_spawn import _DDP_FORK_ALIASES
from lightning_lite.utilities.exceptions import MisconfigurationException


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


@RunIf(skip_windows=True)
def test_strategy_choice_ddp_spawn_on_cpu():
    """Test that selecting DDPSpawnStrategy on CPU works."""
    _test_strategy_choice_ddp_and_cpu(ddp_strategy_class=DDPSpawnStrategy)


def _test_strategy_choice_ddp_and_cpu(ddp_strategy_class):
    connector = _Connector(
        strategy=ddp_strategy_class(find_unused_parameters=True),
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
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=0)
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
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=0)
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
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=0)
def test_dist_backend_accelerator_mapping(*_):
    connector = _Connector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert connector.strategy.local_rank == 0


@RunIf(mps=False)
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_ipython_incompatible_backend_error(_, monkeypatch):
    monkeypatch.setattr(lightning_lite.connector, "_IS_INTERACTIVE", True)
    with pytest.raises(RuntimeError, match=r"strategy='ddp'\)`.*is not compatible"):
        _Connector(strategy="ddp", accelerator="gpu", devices=2)

    with pytest.raises(RuntimeError, match=r"strategy='ddp_spawn'\)`.*is not compatible"):
        _Connector(strategy="ddp_spawn", accelerator="gpu", devices=2)

    with pytest.raises(RuntimeError, match=r"strategy='ddp_sharded_spawn'\)`.*is not compatible"):
        _Connector(strategy="ddp_sharded_spawn", accelerator="gpu", devices=2)

    with pytest.raises(RuntimeError, match=r"strategy='ddp'\)`.*is not compatible"):
        # Edge case: _Connector maps dp to ddp if accelerator != gpu
        _Connector(strategy="dp")


@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_ipython_compatible_dp_strategy_gpu(_, monkeypatch):
    monkeypatch.setattr(lightning_lite.utilities.imports, "_IS_INTERACTIVE", True)
    connector = _Connector(strategy="dp", accelerator="gpu")
    assert connector.strategy.launcher is None


@RunIf(skip_windows=True)
def test_ipython_compatible_strategy_tpu(tpu_available, monkeypatch):
    monkeypatch.setattr(lightning_lite.utilities.imports, "_IS_INTERACTIVE", True)
    connector = _Connector(accelerator="tpu")
    assert connector.strategy.launcher.is_interactive_compatible


@RunIf(skip_windows=True)
def test_ipython_compatible_strategy_ddp_fork(monkeypatch):
    monkeypatch.setattr(lightning_lite.utilities.imports, "_IS_INTERACTIVE", True)
    connector = _Connector(strategy="ddp_fork", accelerator="cpu")
    assert connector.strategy.launcher.is_interactive_compatible


@RunIf(mps=False)
@pytest.mark.parametrize(
    ["strategy", "strategy_class"],
    [
        ("ddp", DDPStrategy),
        ("ddp_spawn", DDPSpawnStrategy),
        ("ddp_sharded", DDPShardedStrategy),
        ("ddp_sharded_spawn", DDPSpawnShardedStrategy),
        pytest.param("deepspeed", DeepSpeedStrategy, marks=RunIf(deepspeed=True)),
    ],
)
@pytest.mark.parametrize("devices", [1, 2])
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_strategy_choice_multi_node_gpu(_, strategy, strategy_class, devices):
    connector = _Connector(num_nodes=2, accelerator="gpu", strategy=strategy, devices=devices)
    assert isinstance(connector.strategy, strategy_class)


@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=0)
def test_cuda_accelerator_can_not_run_on_system(_):
    connector = _Connector(accelerator="cpu")
    assert isinstance(connector.accelerator, CPUAccelerator)

    with pytest.raises(
        RuntimeError,
        match="CUDAAccelerator` can not run on your system since the accelerator is not available.",
    ):
        _Connector(accelerator="cuda", devices=1)


@pytest.mark.skipif(TPUAccelerator.is_available(), reason="test requires missing TPU")
@mock.patch("lightning_lite.accelerators.tpu._XLA_AVAILABLE", True)
def test_tpu_accelerator_can_not_run_on_system():
    with pytest.raises(RuntimeError, match="TPUAccelerator` can not run on your system"):
        _Connector(accelerator="tpu", devices=8)


@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
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


@pytest.mark.parametrize(["devices", "strategy_class"], [(1, SingleDeviceStrategy), (5, DDPSpawnStrategy)])
def test_accelerator_cpu_with_devices(devices, strategy_class):
    connector = _Connector(accelerator="cpu", devices=devices)
    assert connector._parallel_devices == [torch.device("cpu")] * devices
    assert isinstance(connector.strategy, strategy_class)
    assert isinstance(connector.accelerator, CPUAccelerator)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ["devices", "strategy_class"], [(1, SingleDeviceStrategy), ([1], SingleDeviceStrategy), (2, DDPSpawnStrategy)]
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


def test_unsupported_strategy_types_on_cpu_and_fallback():
    with pytest.warns(UserWarning, match="is not supported on CPUs, hence setting `strategy='ddp"):
        connector = _Connector(strategy="dp", devices=2)
    assert isinstance(connector.strategy, DDPStrategy)


def test_invalid_accelerator_choice():
    with pytest.raises(ValueError, match="You selected an invalid accelerator name: `accelerator='cocofruit'`"):
        _Connector(accelerator="cocofruit")


def test_invalid_strategy_choice():
    with pytest.raises(ValueError, match="You selected an invalid strategy name: `strategy='cocofruit'`"):
        _Connector(strategy="cocofruit")


@pytest.mark.parametrize(
    ["strategy", "strategy_class"],
    [
        ("ddp_spawn", DDPSpawnStrategy),
        ("ddp_spawn_find_unused_parameters_false", DDPSpawnStrategy),
        ("ddp", DDPStrategy),
        ("ddp_find_unused_parameters_false", DDPStrategy),
    ],
)
def test_strategy_choice_cpu_str(strategy, strategy_class):
    connector = _Connector(strategy=strategy, accelerator="cpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)


@pytest.mark.parametrize("strategy_class", [DDPSpawnStrategy, DDPStrategy])
def test_strategy_choice_cpu_instance(strategy_class):
    connector = _Connector(strategy=strategy_class(), accelerator="cpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ["strategy", "strategy_class"],
    [
        ("ddp_spawn", DDPSpawnStrategy),
        ("ddp_spawn_find_unused_parameters_false", DDPSpawnStrategy),
        ("ddp", DDPStrategy),
        ("ddp_find_unused_parameters_false", DDPStrategy),
        ("dp", DataParallelStrategy),
        ("ddp_sharded", DDPShardedStrategy),
        ("ddp_sharded_spawn", DDPSpawnShardedStrategy),
        pytest.param("deepspeed", DeepSpeedStrategy, marks=RunIf(deepspeed=True)),
    ],
)
def test_strategy_choice_gpu_str(strategy, strategy_class):
    connector = _Connector(strategy=strategy, accelerator="gpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)


@RunIf(fairscale=True)
@pytest.mark.parametrize(
    "strategy,expected_strategy", [("ddp_sharded", DDPShardedStrategy), ("ddp_sharded_spawn", DDPSpawnShardedStrategy)]
)
@pytest.mark.parametrize(
    "precision,expected_precision", [(16, NativeMixedPrecision), (32, Precision), ("bf16", NativeMixedPrecision)]
)
def test_strategy_choice_sharded(strategy, expected_strategy, precision, expected_precision):
    connector = _Connector(strategy=strategy, devices=1, precision=precision)
    assert isinstance(connector.strategy, expected_strategy)
    assert isinstance(connector.precision, expected_precision)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize("strategy_class", [DDPSpawnStrategy, DDPStrategy])
def test_strategy_choice_gpu_instance(strategy_class):
    connector = _Connector(strategy=strategy_class(), accelerator="gpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize("strategy_class", [DDPSpawnStrategy, DDPStrategy])
def test_device_type_when_strategy_instance_gpu_passed(strategy_class):
    connector = _Connector(strategy=strategy_class(), accelerator="gpu", devices=2)
    assert isinstance(connector.strategy, strategy_class)
    assert isinstance(connector.accelerator, CUDAAccelerator)


@pytest.mark.parametrize("precision", [1, 12, "invalid"])
def test_validate_precision_type(precision):
    with pytest.raises(ValueError, match=f"Precision {repr(precision)} is invalid"):
        _Connector(precision=precision)


def test_strategy_choice_ddp_spawn_cpu():
    connector = _Connector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPSpawnStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy.launcher._start_method == "spawn"


@RunIf(skip_windows=True)
@mock.patch("lightning_lite.connector._IS_INTERACTIVE", True)
def test_strategy_choice_ddp_fork_in_interactive():
    """Test that when accelerator and strategy are unspecified, the connector chooses DDP Fork in interactive
    environments by default."""
    connector = _Connector(devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPSpawnStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy.launcher._start_method == "fork"


@RunIf(skip_windows=True)
def test_strategy_choice_ddp_fork_cpu():
    connector = _Connector(strategy="ddp_fork", accelerator="cpu", devices=2)
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, DDPSpawnStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)
    assert connector.strategy.launcher._start_method == "fork"


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp(*_):
    connector = _Connector(strategy="ddp", accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_spawn(*_):
    connector = _Connector(strategy="ddp_spawn", accelerator="gpu", devices=1)
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPSpawnStrategy)
    assert isinstance(connector.strategy.cluster_environment, LightningEnvironment)


@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
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
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
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
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=1)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
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
def test_unsupported_tpu_choice(tpu_available):
    with pytest.raises(NotImplementedError, match=r"accelerator='tpu', precision=64\)` is not implemented"):
        _Connector(accelerator="tpu", precision=64)

    # if user didn't set strategy, _Connector will choose the TPUSingleStrategy or XLAStrategy
    with pytest.raises(ValueError, match="TPUAccelerator` can only be used with a `SingleTPUStrategy`"), pytest.warns(
        UserWarning, match=r"accelerator='tpu', precision=16\)` but native AMP is not supported"
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


@mock.patch("lightning_lite.accelerators.cuda.CUDAAccelerator.is_available", return_value=False)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_devices_auto_choice_cpu(tpu_available, *_):
    connector = _Connector(accelerator="auto", devices="auto")
    assert isinstance(connector.accelerator, CPUAccelerator)
    assert isinstance(connector.strategy, SingleDeviceStrategy)
    assert connector.strategy.root_device == torch.device("cpu")


@RunIf(mps=False)
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=2)
def test_devices_auto_choice_gpu(*_):
    connector = _Connector(accelerator="auto", devices="auto")
    assert isinstance(connector.accelerator, CUDAAccelerator)
    assert isinstance(connector.strategy, DDPSpawnStrategy)
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


@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning_lite.accelerators.cuda.num_cuda_devices", return_value=1)
def test_gpu_accelerator_backend_choice_cuda(*_):
    connector = _Connector(accelerator="gpu")
    assert connector._accelerator_flag == "cuda"
    assert isinstance(connector.accelerator, CUDAAccelerator)


@RunIf(min_torch="1.12")
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=True)
@mock.patch("lightning_lite.accelerators.mps._get_all_available_mps_gpus", return_value=[0])
def test_gpu_accelerator_backend_choice_mps(*_):
    connector = _Connector(accelerator="gpu")
    assert connector._accelerator_flag == "mps"
    assert isinstance(connector.accelerator, MPSAccelerator)


@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning_lite.accelerators.cuda.CUDAAccelerator.is_available", return_value=False)
def test_gpu_accelerator_no_gpu_backend_found_error(*_):
    with pytest.raises(RuntimeError, match="No supported gpu backend found!"):
        _Connector(accelerator="gpu")


@pytest.mark.parametrize("strategy", _DDP_FORK_ALIASES)
@mock.patch(
    "lightning_lite.connector.torch.multiprocessing.get_all_start_methods",
    return_value=[],
)
def test_ddp_fork_on_unsupported_platform(_, strategy):
    with pytest.raises(ValueError, match="process forking is not supported on this platform"):
        _Connector(strategy=strategy)


@mock.patch("lightning_lite.plugins.precision.native_amp._TORCH_GREATER_EQUAL_1_10", True)
def test_precision_selection_16_on_cpu_warns():
    with pytest.warns(
        UserWarning, match=r"precision=16\)` but native AMP is not supported on CPU. Using `precision='bf16"
    ):
        _Connector(precision=16)


@mock.patch("lightning_lite.plugins.precision.native_amp._TORCH_GREATER_EQUAL_1_10", False)
def test_precision_selection_16_raises_torch_version(monkeypatch):
    with pytest.raises(ImportError, match="must install torch greater or equal to 1.10"):
        _Connector(accelerator="cpu", precision=16)
    with pytest.raises(ImportError, match="must install torch greater or equal to 1.10"):
        _Connector(accelerator="cpu", precision="bf16")


class MyNativeAMP(NativeMixedPrecision):
    pass


@RunIf(mps=False)
@pytest.mark.parametrize("strategy,devices", [("ddp", 2), ("ddp_spawn", 2)])
@pytest.mark.parametrize(
    "is_custom_plugin,plugin_cls",
    [(False, NativeMixedPrecision), (True, MyNativeAMP)],
)
@mock.patch("lightning_lite.plugins.precision.native_amp._TORCH_GREATER_EQUAL_1_10", True)
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


@pytest.mark.parametrize(
    ["strategy", "strategy_cls"], [("DDP", DDPStrategy), ("DDP_FIND_UNUSED_PARAMETERS_FALSE", DDPStrategy)]
)
def test_strategy_str_passed_being_case_insensitive(strategy, strategy_cls):
    connector = _Connector(strategy=strategy)
    assert isinstance(connector.strategy, strategy_cls)
