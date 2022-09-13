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
from lightning_lite.accelerators.accelerator import Accelerator
from lightning_lite.accelerators.cpu import CPUAccelerator
from lightning_lite.accelerators.cuda import CUDAAccelerator
from lightning_lite.accelerators.mps import MPSAccelerator
from lightning_lite.connector import AcceleratorConnector
from lightning_lite.plugins import DoublePrecision, Precision
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
)
from lightning_lite.strategies.ddp_spawn import _DDP_FORK_ALIASES
from lightning_lite.utilities.exceptions import MisconfigurationException


def test_accelerator_choice_cpu(tmpdir):
    trainer = AcceleratorConnector()
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, SingleDeviceStrategy)


def test_accelerator_invalid_choice():
    with pytest.raises(ValueError, match="You selected an invalid accelerator name: `accelerator='invalid'`"):
        AcceleratorConnector(accelerator="invalid")


@RunIf(skip_windows=True, standalone=True)
def test_strategy_choice_ddp_on_cpu(tmpdir):
    """Test that selecting DDPStrategy on CPU works."""
    _test_strategy_choice_ddp_and_cpu(ddp_strategy_class=DDPStrategy)


@RunIf(skip_windows=True)
def test_strategy_choice_ddp_spawn_on_cpu(tmpdir):
    """Test that selecting DDPSpawnStrategy on CPU works."""
    _test_strategy_choice_ddp_and_cpu(ddp_strategy_class=DDPSpawnStrategy)


def _test_strategy_choice_ddp_and_cpu(ddp_strategy_class):
    trainer = AcceleratorConnector(
        strategy=ddp_strategy_class(find_unused_parameters=True),
        accelerator="cpu",
        devices=2,
    )
    assert isinstance(trainer.strategy, ddp_strategy_class)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert trainer.strategy.num_processes == 2
    assert trainer.strategy.parallel_devices == [torch.device("cpu")] * 2


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
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=0)
def test_custom_cluster_environment_in_slurm_environment(_):
    """Test that we choose the custom cluster even when SLURM or TE flags are around."""

    class CustomCluster(LightningEnvironment):
        @property
        def main_address(self):
            return "asdf"

        @property
        def creates_processes_externally(self) -> bool:
            return True

    trainer = AcceleratorConnector(
        plugins=[CustomCluster()],
        accelerator="cpu",
        strategy="ddp",
        devices=2,
    )
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, CustomCluster)


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
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=0)
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

    strategy = Strat(device=torch.device("cpu"), accelerator=Accel(), precision_plugin=Prec())
    trainer = AcceleratorConnector(strategy=strategy, devices=2)
    assert isinstance(trainer.accelerator, Accel)
    assert isinstance(trainer.strategy, Strat)
    assert isinstance(trainer.precision_plugin, Prec)
    assert trainer.strategy is strategy

    class Strat(DDPStrategy):
        pass

    strategy = Strat(accelerator=Accel(), precision_plugin=Prec())
    trainer = AcceleratorConnector(strategy=strategy, devices=2)
    assert isinstance(trainer.accelerator, Accel)
    assert isinstance(trainer.strategy, Strat)
    assert isinstance(trainer.precision_plugin, Prec)
    assert trainer.strategy is strategy


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
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=0)
def test_dist_backend_accelerator_mapping(*_):
    trainer = AcceleratorConnector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert trainer.strategy.local_rank == 0


@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
def test_ipython_incompatible_backend_error(_, monkeypatch):
    monkeypatch.setattr(lightning_lite.utilities, "_IS_INTERACTIVE", True)
    with pytest.raises(MisconfigurationException, match=r"strategy='ddp'\)`.*is not compatible"):
        AcceleratorConnector(strategy="ddp", accelerator="gpu", devices=2)

    with pytest.raises(MisconfigurationException, match=r"strategy='ddp_spawn'\)`.*is not compatible"):
        AcceleratorConnector(strategy="ddp_spawn", accelerator="gpu", devices=2)

    with pytest.raises(MisconfigurationException, match=r"strategy='ddp_sharded_spawn'\)`.*is not compatible"):
        AcceleratorConnector(strategy="ddp_sharded_spawn", accelerator="gpu", devices=2)

    with pytest.raises(MisconfigurationException, match=r"strategy='ddp'\)`.*is not compatible"):
        # Edge case: AcceleratorConnector maps dp to ddp if accelerator != gpu
        AcceleratorConnector(strategy="dp")


@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
def test_ipython_compatible_dp_strategy_gpu(_, monkeypatch):
    monkeypatch.setattr(lightning_lite.utilities, "_IS_INTERACTIVE", True)
    trainer = AcceleratorConnector(strategy="dp", accelerator="gpu")
    assert trainer.strategy.launcher is None


@RunIf(skip_windows=True)
@mock.patch("lightning_lite.accelerators.tpu.TPUAccelerator.is_available", return_value=True)
def test_ipython_compatible_strategy_tpu(_, monkeypatch):
    monkeypatch.setattr(lightning_lite.utilities, "_IS_INTERACTIVE", True)
    trainer = AcceleratorConnector(accelerator="tpu")
    assert trainer.strategy.launcher.is_interactive_compatible


@RunIf(skip_windows=True)
def test_ipython_compatible_strategy_ddp_fork(monkeypatch):
    monkeypatch.setattr(lightning_lite.utilities, "_IS_INTERACTIVE", True)
    trainer = AcceleratorConnector(strategy="ddp_fork", accelerator="cpu")
    assert trainer.strategy.launcher.is_interactive_compatible


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
@mock.patch("lightning_lite.utilities.device_parser.is_cuda_available", return_value=True)
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
@mock.patch("lightning_lite.utilities.device_parser._get_all_available_mps_gpus", return_value=[0, 1])
def test_accelerator_choice_multi_node_gpu(_, __, ___, strategy, strategy_class, devices):
    trainer = AcceleratorConnector(num_nodes=2, accelerator="gpu", strategy=strategy, devices=devices)
    assert isinstance(trainer.strategy, strategy_class)


@mock.patch("lightning_lite.accelerators.cuda.device_parser.num_cuda_devices", return_value=0)
def test_accelerator_cpu(*_):
    trainer = AcceleratorConnector(accelerator="cpu")
    assert isinstance(trainer.accelerator, CPUAccelerator)

    with pytest.raises(
        RuntimeError,
        match="CUDAAccelerator can not run on your system since the accelerator is not available",
    ):
        with pytest.deprecated_call(match=r"is deprecated in v1.7 and will be removed"):
            AcceleratorConnector(gpus=1)

    with pytest.raises(
        RuntimeError,
        match="CUDAAccelerator can not run on your system since the accelerator is not available.",
    ):
        AcceleratorConnector(accelerator="cuda")

    with pytest.deprecated_call(match=r"is deprecated in v1.7 and will be removed"):
        AcceleratorConnector(accelerator="cpu", gpus=1)


@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
@mock.patch("lightning_lite.utilities.device_parser.is_cuda_available", return_value=True)
@pytest.mark.parametrize("device_count", (["0"], [0, "1"], ["GPU"], [["0", "1"], [0, 1]], [False]))
def test_accelererator_invalid_type_devices(_, __, device_count):
    with pytest.raises(
        MisconfigurationException, match=r"must be an int, a string, a sequence of ints or None, but you"
    ):
        _ = AcceleratorConnector(accelerator="gpu", devices=device_count)


@RunIf(min_cuda_gpus=1)
def test_accelerator_gpu():
    trainer = AcceleratorConnector(accelerator="gpu", devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)

    trainer = AcceleratorConnector(accelerator="gpu")
    assert isinstance(trainer.accelerator, CUDAAccelerator)

    trainer = AcceleratorConnector(accelerator="auto", devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)


@pytest.mark.parametrize(["devices", "strategy_class"], [(1, SingleDeviceStrategy), (5, DDPSpawnStrategy)])
def test_accelerator_cpu_with_devices(devices, strategy_class):
    trainer = AcceleratorConnector(accelerator="cpu", devices=devices)
    assert trainer._parallel_devices == [torch.device("cpu")] * devices
    assert isinstance(trainer.strategy, strategy_class)
    assert isinstance(trainer.accelerator, CPUAccelerator)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize(
    ["devices", "strategy_class"], [(1, SingleDeviceStrategy), ([1], SingleDeviceStrategy), (2, DDPSpawnStrategy)]
)
def test_accelerator_gpu_with_devices(devices, strategy_class):
    trainer = AcceleratorConnector(accelerator="gpu", devices=devices)
    assert len(trainer._parallel_devices) == len(devices) if isinstance(devices, list) else devices
    assert isinstance(trainer.strategy, strategy_class)
    assert isinstance(trainer.accelerator, CUDAAccelerator)


@RunIf(min_cuda_gpus=1)
def test_accelerator_auto_with_devices_gpu():
    trainer = AcceleratorConnector(accelerator="auto", devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert trainer._parallel_devices == [torch.device("cuda", 0)]


def test_set_devices_if_none_cpu():
    trainer = AcceleratorConnector(accelerator="cpu", devices=3)
    assert trainer._parallel_devices == [torch.device("cpu")] * 3


def test_unsupported_strategy_types_on_cpu_and_fallback():
    with pytest.warns(UserWarning, match="is not supported on CPUs, hence setting `strategy='ddp"):
        trainer = AcceleratorConnector(strategy="dp", devices=2)
    assert isinstance(trainer.strategy, DDPStrategy)


# TODO(lite): This error handling needs to be fixed
def test_exception_invalid_strategy():
    with pytest.raises(ValueError, match=r"strategy='ddp_cpu'\)` is not a valid"):
        AcceleratorConnector(strategy="ddp_cpu")
    with pytest.raises(ValueError, match=r"strategy='tpu_spawn'\)` is not a valid"):
        AcceleratorConnector(strategy="tpu_spawn")


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
    trainer = AcceleratorConnector(strategy=strategy, accelerator="cpu", devices=2)
    assert isinstance(trainer.strategy, strategy_class)


@pytest.mark.parametrize("strategy_class", [DDPSpawnStrategy, DDPStrategy])
def test_strategy_choice_cpu_instance(strategy_class):
    trainer = AcceleratorConnector(strategy=strategy_class(), accelerator="cpu", devices=2)
    assert isinstance(trainer.strategy, strategy_class)


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
    trainer = AcceleratorConnector(strategy=strategy, accelerator="gpu", devices=2)
    assert isinstance(trainer.strategy, strategy_class)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize("strategy_class", [DDPSpawnStrategy, DDPStrategy])
def test_strategy_choice_gpu_instance(strategy_class):
    trainer = AcceleratorConnector(strategy=strategy_class(), accelerator="gpu", devices=2)
    assert isinstance(trainer.strategy, strategy_class)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize("strategy_class", [DDPSpawnStrategy, DDPStrategy])
def test_device_type_when_strategy_instance_gpu_passed(strategy_class):
    trainer = AcceleratorConnector(strategy=strategy_class(), accelerator="gpu", devices=2)
    assert isinstance(trainer.strategy, strategy_class)
    assert isinstance(trainer.accelerator, CUDAAccelerator)


@pytest.mark.parametrize("precision", [1, 12, "invalid"])
def test_validate_precision_type(precision):
    with pytest.raises(ValueError, match=f"Precision {repr(precision)} is invalid"):
        AcceleratorConnector(precision=precision)


def test_strategy_choice_ddp_spawn_cpu():
    trainer = AcceleratorConnector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPSpawnStrategy)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)
    assert trainer.strategy.launcher._start_method == "spawn"


@RunIf(skip_windows=True)
@mock.patch("lightning_lite.connector._IS_INTERACTIVE", True)
def test_strategy_choice_ddp_fork_in_interactive():
    """Test that when accelerator and strategy are unspecified, the connector chooses DDP Fork in interactive
    environments by default."""
    trainer = AcceleratorConnector(devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPSpawnStrategy)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)
    assert trainer.strategy.launcher._start_method == "fork"


@RunIf(skip_windows=True)
def test_strategy_choice_ddp_fork_cpu():
    trainer = AcceleratorConnector(strategy="ddp_fork", accelerator="cpu", devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPSpawnStrategy)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)
    assert trainer.strategy.launcher._start_method == "fork"


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
@mock.patch("lightning_lite.utilities.device_parser.is_cuda_available", return_value=True)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp(*_):
    trainer = AcceleratorConnector(strategy="ddp", accelerator="gpu", devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
@mock.patch("lightning_lite.utilities.device_parser.is_cuda_available", return_value=True)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_spawn(*_):
    trainer = AcceleratorConnector(strategy="ddp_spawn", accelerator="gpu", devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, DDPSpawnStrategy)
    assert isinstance(trainer.strategy.cluster_environment, LightningEnvironment)


@RunIf(min_cuda_gpus=2)
@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "SLURM_PROCID": "1",
        "SLURM_LOCALID": "1",
    },
)
@mock.patch("pytorch_lightning.strategies.DDPStrategy.setup_distributed", autospec=True)
@pytest.mark.parametrize("strategy", ["ddp", DDPStrategy()])
def test_strategy_choice_ddp_slurm(_, strategy):
    trainer = AcceleratorConnector(strategy=strategy, accelerator="gpu", devices=2)
    assert trainer._is_slurm_managing_tasks()
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, SLURMEnvironment)
    assert trainer.strategy.cluster_environment.local_rank() == 1
    assert trainer.strategy.local_rank == 1


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
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
@mock.patch("lightning_lite.utilities.device_parser.is_cuda_available", return_value=True)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_te(*_):
    trainer = AcceleratorConnector(strategy="ddp", accelerator="gpu", devices=2)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, TorchElasticEnvironment)
    assert trainer.strategy.cluster_environment.local_rank() == 1
    assert trainer.strategy.local_rank == 1


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
    trainer = AcceleratorConnector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, TorchElasticEnvironment)
    assert trainer.strategy.cluster_environment.local_rank() == 1
    assert trainer.strategy.local_rank == 1


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
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=1)
@mock.patch("lightning_lite.utilities.device_parser.is_cuda_available", return_value=True)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_strategy_choice_ddp_kubeflow(*_):
    trainer = AcceleratorConnector(strategy="ddp", accelerator="gpu", devices=1)
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, KubeflowEnvironment)
    assert trainer.strategy.cluster_environment.local_rank() == 0
    assert trainer.strategy.local_rank == 0


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
    trainer = AcceleratorConnector(strategy="ddp_spawn", accelerator="cpu", devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, KubeflowEnvironment)
    assert trainer.strategy.cluster_environment.local_rank() == 0
    assert trainer.strategy.local_rank == 0


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
    trainer = AcceleratorConnector(strategy=strategy, accelerator="cpu", devices=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, DDPStrategy)
    assert isinstance(trainer.strategy.cluster_environment, SLURMEnvironment)
    assert trainer.strategy.local_rank == 0


@mock.patch("lightning_lite.accelerators.tpu.TPUAccelerator.is_available", return_value=True)
def test_unsupported_tpu_choice(_):

    with pytest.raises(NotImplementedError, match=r"accelerator='tpu', precision=64\)` is not implemented"):
        AcceleratorConnector(accelerator="tpu", precision=64)

    # if user didn't set strategy, AcceleratorConnector will choose the TPUSingleStrategy or TPUSpawnStrategy
    with pytest.raises(ValueError, match="TPUAccelerator` can only be used with a `SingleTPUStrategy`"):
        with pytest.warns(UserWarning, match=r"accelerator='tpu', precision=16\)` but native AMP is not supported"):
            AcceleratorConnector(accelerator="tpu", precision=16, strategy="ddp")


@mock.patch("lightning_lite.accelerators.cuda.CUDAAccelerator.is_available", return_value=False)
@mock.patch("lightning_lite.accelerators.tpu.TPUAccelerator.is_available", return_value=False)
@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
def test_devices_auto_choice_cpu(*_):
    trainer = AcceleratorConnector(accelerator="auto", devices="auto")
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer.strategy.root_device == torch.device("cpu")


@RunIf(mps=False)
@mock.patch("lightning_lite.utilities.device_parser.is_cuda_available", return_value=True)
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=2)
def test_devices_auto_choice_gpu(*_):
    trainer = AcceleratorConnector(accelerator="auto", devices="auto")
    assert isinstance(trainer.accelerator, CUDAAccelerator)
    assert isinstance(trainer.strategy, DDPSpawnStrategy)
    assert len(trainer._parallel_devices) == 2


@RunIf(mps=True)
def test_devices_auto_choice_mps():
    trainer = AcceleratorConnector(accelerator="auto", devices="auto")
    assert isinstance(trainer.accelerator, MPSAccelerator)
    assert isinstance(trainer.strategy, SingleDeviceStrategy)
    assert trainer.strategy.root_device == torch.device("mps", 0)
    assert trainer._parallel_devices == [torch.device("mps", 0)]


@pytest.mark.parametrize(
    ["parallel_devices", "accelerator"],
    [([torch.device("cpu")], "cuda"), ([torch.device("cuda", i) for i in range(8)], "tpu")],
)
def test_parallel_devices_in_strategy_confilict_with_accelerator(parallel_devices, accelerator):
    with pytest.raises(ValueError, match=r"parallel_devices set through"):
        AcceleratorConnector(strategy=DDPStrategy(parallel_devices=parallel_devices), accelerator=accelerator)


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
        AcceleratorConnector(plugins=plugins)


@pytest.mark.parametrize("accelerator", ("cpu", "cuda", "mps", "tpu"))
@pytest.mark.parametrize("devices", ("0", 0, []))
def test_passing_zero_and_empty_list_to_devices_flag(accelerator, devices):
    with pytest.raises(ValueError, match="value is not a valid input using"):
        AcceleratorConnector(accelerator=accelerator, devices=devices)


@pytest.mark.parametrize(
    "expected_accelerator_flag,expected_accelerator_class",
    [
        pytest.param("cuda", CUDAAccelerator, marks=RunIf(min_cuda_gpus=1)),
        pytest.param("mps", MPSAccelerator, marks=RunIf(mps=True)),
    ],
)
def test_gpu_accelerator_backend_choice(expected_accelerator_flag, expected_accelerator_class):
    trainer = AcceleratorConnector(accelerator="gpu")
    assert trainer._accelerator_flag == expected_accelerator_flag
    assert isinstance(trainer.accelerator, expected_accelerator_class)


@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning_lite.utilities.device_parser.num_cuda_devices", return_value=1)
def test_gpu_accelerator_backend_choice_cuda(*_):
    trainer = AcceleratorConnector(accelerator="gpu")
    assert trainer._accelerator_flag == "cuda"
    assert isinstance(trainer.accelerator, CUDAAccelerator)


@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=True)
@mock.patch("lightning_lite.utilities.device_parser._get_all_available_mps_gpus", return_value=[0])
@mock.patch("torch.device", return_value="mps")  # necessary because torch doesn't allow creation of mps devices
def test_gpu_accelerator_backend_choice_mps(*_):
    trainer = AcceleratorConnector(accelerator="gpu")
    assert trainer._accelerator_flag == "mps"
    assert isinstance(trainer.accelerator, MPSAccelerator)


@mock.patch("lightning_lite.accelerators.mps.MPSAccelerator.is_available", return_value=False)
@mock.patch("lightning_lite.accelerators.cuda.CUDAAccelerator.is_available", return_value=False)
def test_gpu_accelerator_no_gpu_backend_found_error(*_):
    with pytest.raises(RuntimeError, match="No supported gpu backend found!"):
        AcceleratorConnector(accelerator="gpu")


@pytest.mark.parametrize("strategy", _DDP_FORK_ALIASES)
@mock.patch(
    "lightning_lite.connector.torch.multiprocessing.get_all_start_methods",
    return_value=[],
)
def test_ddp_fork_on_unsupported_platform(_, strategy):
    with pytest.raises(ValueError, match="process forking is not supported on this platform"):
        AcceleratorConnector(strategy=strategy)


@RunIf(skip_windows=True)
@pytest.mark.parametrize("strategy", _DDP_FORK_ALIASES)
@mock.patch.dict(os.environ, {"PL_DISABLE_FORK": "1"}, clear=True)
def test_strategy_choice_ddp_spawn_in_interactive_when_fork_disabled(strategy):
    """Test there is an error when forking is disabled via the environment variable and the user requests fork."""
    with pytest.raises(ValueError, match="Forking is disabled in this environment"):
        AcceleratorConnector(devices=2, strategy=strategy)
