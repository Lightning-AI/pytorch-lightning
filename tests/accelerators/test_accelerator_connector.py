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
from typing import Optional
from unittest import mock

import pytest
import torch
import torch.distributed

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.gpu import GPUAccelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import (
    DataParallelPlugin,
    DDP2Plugin,
    DDPPlugin,
    DDPShardedPlugin,
    DDPSpawnPlugin,
    DDPSpawnShardedPlugin,
    DeepSpeedPlugin,
    ParallelPlugin,
    PrecisionPlugin,
    SingleDevicePlugin,
)
from pytorch_lightning.plugins.environments import (
    KubeflowEnvironment,
    LightningEnvironment,
    SLURMEnvironment,
    TorchElasticEnvironment,
)
from pytorch_lightning.utilities import DeviceType, DistributedType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


def test_accelerator_choice_cpu(tmpdir):
    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.training_type_plugin, SingleDevicePlugin)


@pytest.mark.parametrize(("num_processes", "num_nodes"), ([(1, 1), (1, 2), (2, 1), (2, 2)]))
def test_accelerator_choice_ddp_cpu(tmpdir, num_processes: int, num_nodes: int):
    trainer = Trainer(fast_dev_run=True, accelerator="ddp_cpu", num_processes=num_processes, num_nodes=num_nodes)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    no_spawn = num_processes == 1 and num_nodes > 1
    assert isinstance(trainer.training_type_plugin, DDPPlugin if no_spawn else DDPSpawnPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("torch.cuda.is_available", return_value=True)
def test_accelerator_choice_ddp(cuda_available_mock, device_count_mock):
    with pytest.deprecated_call(match=r"accelerator='ddp'\)` has been deprecated"):
        trainer = Trainer(fast_dev_run=True, accelerator="ddp", gpus=1)
    assert isinstance(trainer.accelerator, GPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("torch.cuda.is_available", return_value=True)
def test_accelerator_choice_ddp_spawn(cuda_available_mock, device_count_mock):
    with pytest.deprecated_call(match=r"accelerator='ddp_spawn'\)` has been deprecated"):
        trainer = Trainer(fast_dev_run=True, accelerator="ddp_spawn", gpus=1)
    assert isinstance(trainer.accelerator, GPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPSpawnPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@RunIf(min_gpus=2)
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
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_accelerator_choice_ddp_slurm(setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer._accelerator_connector._is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, accelerator="ddp", gpus=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=2)
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
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_accelerator_choice_ddp2_slurm(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer._accelerator_connector._is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDP2Plugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, accelerator="ddp2", gpus=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1)
@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_accelerator_choice_ddp_te(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, accelerator="ddp", gpus=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1)
@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_accelerator_choice_ddp2_te(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDP2Plugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, accelerator="ddp2", gpus=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(
    os.environ, {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1", "GROUP_RANK": "0"}
)
@mock.patch("torch.cuda.device_count", return_value=0)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_accelerator_choice_ddp_cpu_te(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, accelerator="ddp_cpu", num_processes=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1)
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
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_accelerator_choice_ddp_kubeflow(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, KubeflowEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 0
            assert trainer.training_type_plugin.local_rank == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, accelerator="ddp", gpus=1, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


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
@mock.patch("torch.cuda.device_count", return_value=0)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_accelerator_choice_ddp_cpu_kubeflow(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, KubeflowEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 0
            assert trainer.training_type_plugin.local_rank == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, accelerator="ddp_cpu", num_processes=1, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


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
@mock.patch("torch.cuda.device_count", return_value=0)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_accelerator_choice_ddp_cpu_slurm(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer._accelerator_connector._is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.local_rank == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, accelerator="ddp_cpu", num_processes=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(special=True)
def test_accelerator_choice_ddp_cpu_and_plugin(tmpdir):
    """Test that accelerator="ddp_cpu" can work together with an instance of DDPPlugin."""
    _test_accelerator_choice_ddp_cpu_and_plugin(tmpdir, ddp_plugin_class=DDPPlugin)


@RunIf(special=True)
def test_accelerator_choice_ddp_cpu_and_plugin_spawn(tmpdir):
    """Test that accelerator="ddp_cpu" can work together with an instance of DDPPSpawnPlugin."""
    _test_accelerator_choice_ddp_cpu_and_plugin(tmpdir, ddp_plugin_class=DDPSpawnPlugin)


def _test_accelerator_choice_ddp_cpu_and_plugin(tmpdir, ddp_plugin_class):
    trainer = Trainer(
        default_root_dir=tmpdir,
        plugins=[ddp_plugin_class(find_unused_parameters=True)],
        fast_dev_run=True,
        accelerator="ddp_cpu",
        num_processes=2,
    )
    assert isinstance(trainer.training_type_plugin, ddp_plugin_class)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert trainer.training_type_plugin.num_processes == 2
    assert trainer.training_type_plugin.parallel_devices == [torch.device("cpu")] * 2


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
@mock.patch("torch.cuda.device_count", return_value=0)
def test_accelerator_choice_ddp_cpu_custom_cluster(_, tmpdir):
    """Test that we choose the custom cluster even when SLURM or TE flags are around."""

    class CustomCluster(LightningEnvironment):
        def master_address(self):
            return "asdf"

        @property
        def creates_processes_externally(self) -> bool:
            return True

    trainer = Trainer(
        default_root_dir=tmpdir, plugins=[CustomCluster()], fast_dev_run=True, accelerator="ddp_cpu", num_processes=2
    )
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, CustomCluster)


@mock.patch.dict(
    os.environ,
    {"SLURM_NTASKS": "2", "SLURM_JOB_NAME": "SOME_NAME", "SLURM_NODEID": "0", "LOCAL_RANK": "0", "SLURM_LOCALID": "0"},
)
@mock.patch("torch.cuda.device_count", return_value=0)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_custom_accelerator(device_count_mock, setup_distributed_mock):
    class Accel(Accelerator):
        pass

    class Prec(PrecisionPlugin):
        pass

    class TrainTypePlugin(SingleDevicePlugin):
        pass

    ttp = TrainTypePlugin(device=torch.device("cpu"))
    accelerator = Accel(training_type_plugin=ttp, precision_plugin=Prec())
    trainer = Trainer(accelerator=accelerator, fast_dev_run=True, num_processes=2)
    assert isinstance(trainer.accelerator, Accel)
    assert isinstance(trainer.training_type_plugin, TrainTypePlugin)
    assert isinstance(trainer.precision_plugin, Prec)
    assert trainer._accelerator_connector.training_type_plugin is ttp

    class DistributedPlugin(DDPPlugin):
        pass

    ttp = DistributedPlugin()
    accelerator = Accel(training_type_plugin=ttp, precision_plugin=Prec())
    trainer = Trainer(accelerator=accelerator, fast_dev_run=True, num_processes=2)
    assert isinstance(trainer.accelerator, Accel)
    assert isinstance(trainer.training_type_plugin, DistributedPlugin)
    assert isinstance(trainer.precision_plugin, Prec)
    assert trainer._accelerator_connector.training_type_plugin is ttp


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
@mock.patch("torch.cuda.device_count", return_value=0)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_dist_backend_accelerator_mapping(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert trainer.training_type_plugin.local_rank == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy="ddp_spawn", num_processes=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch("pytorch_lightning.utilities._IS_INTERACTIVE", return_value=True)
@mock.patch("torch.cuda.device_count", return_value=2)
def test_ipython_incompatible_backend_error(*_):
    with pytest.raises(MisconfigurationException, match=r"strategy='ddp'\)`.*is not compatible"):
        Trainer(strategy="ddp", gpus=2)

    with pytest.raises(MisconfigurationException, match=r"strategy='ddp2'\)`.*is not compatible"):
        Trainer(strategy="ddp2", gpus=2)


@mock.patch("pytorch_lightning.utilities._IS_INTERACTIVE", return_value=True)
def test_ipython_compatible_backend(*_):
    Trainer(strategy="ddp_spawn", num_processes=2)


@pytest.mark.parametrize(["accelerator", "plugin"], [("ddp_spawn", "ddp_sharded"), (None, "ddp_sharded")])
def test_plugin_accelerator_choice(accelerator: Optional[str], plugin: str):
    """Ensure that when a plugin and accelerator is passed in, that the plugin takes precedent."""
    if accelerator is None:
        with pytest.deprecated_call(match="Passing .* `strategy` to the `plugins`"):
            trainer = Trainer(accelerator=accelerator, plugins=plugin, num_processes=2)
    else:
        with pytest.deprecated_call(match=r"accelerator=.*\)` has been deprecated"):
            trainer = Trainer(accelerator=accelerator, plugins=plugin, num_processes=2)
    assert isinstance(trainer.accelerator.training_type_plugin, DDPShardedPlugin)

    with pytest.deprecated_call(match="Passing .* `strategy` to the `plugins`"):
        trainer = Trainer(plugins=plugin, num_processes=2)
    assert isinstance(trainer.accelerator.training_type_plugin, DDPShardedPlugin)


@pytest.mark.parametrize(
    ["accelerator", "plugin"],
    [
        ("ddp", DDPPlugin),
        ("ddp_spawn", DDPSpawnPlugin),
        ("ddp_sharded", DDPShardedPlugin),
        ("ddp_sharded_spawn", DDPSpawnShardedPlugin),
        pytest.param("deepspeed", DeepSpeedPlugin, marks=RunIf(deepspeed=True)),
    ],
)
@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.device_count", return_value=2)
@pytest.mark.parametrize("gpus", [1, 2])
def test_accelerator_choice_multi_node_gpu(
    mock_is_available, mock_device_count, tmpdir, accelerator: str, plugin: ParallelPlugin, gpus: int
):
    with pytest.deprecated_call(match=r"accelerator=.*\)` has been deprecated"):
        trainer = Trainer(accelerator=accelerator, default_root_dir=tmpdir, num_nodes=2, gpus=gpus)
    assert isinstance(trainer.training_type_plugin, plugin)


@pytest.mark.skipif(torch.cuda.is_available(), reason="test doesn't require GPU")
def test_accelerator_cpu():

    trainer = Trainer(accelerator="cpu")

    assert trainer._device_type == "cpu"
    assert isinstance(trainer.accelerator, CPUAccelerator)

    with pytest.raises(MisconfigurationException, match="You passed `accelerator='gpu'`, but GPUs are not available"):
        trainer = Trainer(accelerator="gpu")

    with pytest.raises(MisconfigurationException, match="You requested GPUs:"):
        trainer = Trainer(accelerator="cpu", gpus=1)


@RunIf(min_gpus=1)
def test_accelerator_gpu():

    trainer = Trainer(accelerator="gpu", gpus=1)

    assert trainer._device_type == "gpu"
    assert isinstance(trainer.accelerator, GPUAccelerator)

    with pytest.raises(
        MisconfigurationException, match="You passed `accelerator='gpu'`, but you didn't pass `gpus` to `Trainer`"
    ):
        trainer = Trainer(accelerator="gpu")

    trainer = Trainer(accelerator="auto", gpus=1)

    assert trainer._device_type == "gpu"
    assert isinstance(trainer.accelerator, GPUAccelerator)


@RunIf(min_gpus=1)
def test_accelerator_cpu_with_gpus_flag():

    trainer = Trainer(accelerator="cpu", gpus=1)

    assert trainer._device_type == "cpu"
    assert isinstance(trainer.accelerator, CPUAccelerator)


@RunIf(min_gpus=2)
def test_accelerator_cpu_with_multiple_gpus():

    trainer = Trainer(accelerator="cpu", gpus=2)

    assert trainer._device_type == "cpu"
    assert isinstance(trainer.accelerator, CPUAccelerator)


@pytest.mark.parametrize(["devices", "plugin"], [(1, SingleDevicePlugin), (5, DDPSpawnPlugin)])
def test_accelerator_cpu_with_devices(devices, plugin):

    trainer = Trainer(accelerator="cpu", devices=devices)

    assert trainer.num_processes == devices
    assert isinstance(trainer.training_type_plugin, plugin)
    assert isinstance(trainer.accelerator, CPUAccelerator)


def test_accelerator_cpu_with_num_processes_priority():
    """Test for checking num_processes takes priority over devices."""

    num_processes = 5
    with pytest.warns(UserWarning, match="The flag `devices=8` will be ignored,"):
        trainer = Trainer(accelerator="cpu", devices=8, num_processes=num_processes)

    assert trainer.num_processes == num_processes


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    ["devices", "plugin"], [(1, SingleDevicePlugin), ([1], SingleDevicePlugin), (2, DDPSpawnPlugin)]
)
def test_accelerator_gpu_with_devices(devices, plugin):

    trainer = Trainer(accelerator="gpu", devices=devices)

    assert trainer.gpus == devices
    assert isinstance(trainer.training_type_plugin, plugin)
    assert isinstance(trainer.accelerator, GPUAccelerator)


@RunIf(min_gpus=1)
def test_accelerator_auto_with_devices_gpu():

    trainer = Trainer(accelerator="auto", devices=1)

    assert trainer._device_type == "gpu"
    assert trainer.gpus == 1


@RunIf(min_gpus=1)
def test_accelerator_gpu_with_gpus_priority():
    """Test for checking `gpus` flag takes priority over `devices`."""

    gpus = 1
    with pytest.warns(UserWarning, match="The flag `devices=4` will be ignored,"):
        trainer = Trainer(accelerator="gpu", devices=4, gpus=gpus)

    assert trainer.gpus == gpus


def test_validate_accelerator_and_devices():

    with pytest.raises(MisconfigurationException, match="You passed `devices=2` but haven't specified"):
        Trainer(accelerator="ddp_cpu", devices=2)


def test_set_devices_if_none_cpu():

    trainer = Trainer(accelerator="cpu", num_processes=3)
    assert trainer.devices == 3


@RunIf(min_gpus=2)
def test_set_devices_if_none_gpu():

    trainer = Trainer(accelerator="gpu", gpus=2)
    assert trainer.devices == 2


def test_devices_with_cpu_only_supports_integer():

    with pytest.raises(MisconfigurationException, match="The flag `devices` must be an int"):
        Trainer(accelerator="cpu", devices="1,3")


@pytest.mark.parametrize("training_type", ["ddp2", "dp"])
def test_unsupported_distrib_types_on_cpu(training_type):

    with pytest.warns(UserWarning, match="is not supported on CPUs, hence setting `strategy='ddp"):
        trainer = Trainer(accelerator=training_type, num_processes=2)

    assert trainer._distrib_type == DistributedType.DDP


def test_accelerator_ddp_for_cpu(tmpdir):
    with pytest.deprecated_call(match=r"accelerator='ddp'\)` has been deprecated"):
        trainer = Trainer(accelerator="ddp", num_processes=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPPlugin)


def test_exception_when_strategy_used_with_accelerator():
    with pytest.raises(MisconfigurationException, match="but have also passed"), pytest.deprecated_call(
        match=r"accelerator='ddp'\)` has been deprecated"
    ):
        Trainer(accelerator="ddp", strategy="ddp_spawn")


def test_exception_when_strategy_used_with_plugins():
    with pytest.raises(MisconfigurationException, match="only specify one training type plugin, but you have passed"):
        Trainer(plugins="ddp_find_unused_parameters_false", strategy="ddp_spawn")


def test_exception_invalid_strategy():
    with pytest.raises(MisconfigurationException, match=r"strategy='ddp_cpu'\)` is not a valid"):
        Trainer(strategy="ddp_cpu")
    with pytest.raises(MisconfigurationException, match=r"strategy='tpu_spawn'\)` is not a valid"):
        Trainer(strategy="tpu_spawn")


@pytest.mark.parametrize(
    ["strategy", "plugin"],
    [
        ("ddp_spawn", DDPSpawnPlugin),
        ("ddp_spawn_find_unused_parameters_false", DDPSpawnPlugin),
        ("ddp", DDPPlugin),
        ("ddp_find_unused_parameters_false", DDPPlugin),
    ],
)
def test_strategy_choice_cpu_str(tmpdir, strategy, plugin):
    trainer = Trainer(strategy=strategy, accelerator="cpu", devices=2)
    assert isinstance(trainer.training_type_plugin, plugin)


@pytest.mark.parametrize("plugin", [DDPSpawnPlugin, DDPPlugin])
def test_strategy_choice_cpu_plugin(tmpdir, plugin):
    trainer = Trainer(strategy=plugin(), accelerator="cpu", devices=2)
    assert isinstance(trainer.training_type_plugin, plugin)


@RunIf(min_gpus=2)
@pytest.mark.parametrize(
    ["strategy", "plugin"],
    [
        ("ddp_spawn", DDPSpawnPlugin),
        ("ddp_spawn_find_unused_parameters_false", DDPSpawnPlugin),
        ("ddp", DDPPlugin),
        ("ddp_find_unused_parameters_false", DDPPlugin),
        ("ddp2", DDP2Plugin),
        ("dp", DataParallelPlugin),
        ("ddp_sharded", DDPShardedPlugin),
        ("ddp_sharded_spawn", DDPSpawnShardedPlugin),
        pytest.param("deepspeed", DeepSpeedPlugin, marks=RunIf(deepspeed=True)),
    ],
)
def test_strategy_choice_gpu_str(tmpdir, strategy, plugin):
    trainer = Trainer(strategy=strategy, accelerator="gpu", devices=2)
    assert isinstance(trainer.training_type_plugin, plugin)


@RunIf(min_gpus=2)
@pytest.mark.parametrize("plugin", [DDPSpawnPlugin, DDPPlugin])
def test_strategy_choice_gpu_plugin(tmpdir, plugin):
    trainer = Trainer(strategy=plugin(), accelerator="gpu", devices=2)
    assert isinstance(trainer.training_type_plugin, plugin)


@RunIf(min_gpus=2)
@pytest.mark.parametrize("plugin", [DDPSpawnPlugin, DDPPlugin])
def test_device_type_when_training_plugin_gpu_passed(tmpdir, plugin):

    trainer = Trainer(strategy=plugin(), gpus=2)
    assert isinstance(trainer.training_type_plugin, plugin)
    assert trainer._device_type == DeviceType.GPU
    assert isinstance(trainer.accelerator, GPUAccelerator)


@pytest.mark.parametrize("precision", [1, 12, "invalid"])
def test_validate_precision_type(tmpdir, precision):

    with pytest.raises(MisconfigurationException, match=f"Precision {repr(precision)} is invalid"):
        Trainer(precision=precision)


def test_amp_level_raises_error_with_native():
    with pytest.raises(MisconfigurationException, match="O2'` but it's only supported with `amp_backend='apex'`"):
        _ = Trainer(amp_level="O2", amp_backend="native", precision=16)


def test_strategy_choice_ddp_spawn_cpu(tmpdir):
    trainer = Trainer(fast_dev_run=True, strategy="ddp_spawn", num_processes=2)
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPSpawnPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("torch.cuda.is_available", return_value=True)
def test_strategy_choice_ddp(cuda_available_mock, device_count_mock):
    trainer = Trainer(fast_dev_run=True, strategy="ddp", gpus=1)
    assert isinstance(trainer.accelerator, GPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("torch.cuda.is_available", return_value=True)
def test_strategy_choice_ddp_spawn(cuda_available_mock, device_count_mock):
    trainer = Trainer(fast_dev_run=True, strategy="ddp_spawn", gpus=1)
    assert isinstance(trainer.accelerator, GPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPSpawnPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@RunIf(min_gpus=2)
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
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
@pytest.mark.parametrize("strategy", ["ddp", DDPPlugin()])
def test_strategy_choice_ddp_slurm(setup_distributed_mock, strategy):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer._accelerator_connector._is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy=strategy, gpus=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=2)
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
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
@pytest.mark.parametrize("strategy", ["ddp2", DDP2Plugin()])
def test_strategy_choice_ddp2_slurm(device_count_mock, setup_distributed_mock, strategy):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer._accelerator_connector._is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDP2Plugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy=strategy, gpus=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1)
@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_strategy_choice_ddp_te(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy="ddp", gpus=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1)
@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
    },
)
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_strategy_choice_ddp2_te(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDP2Plugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy="ddp2", gpus=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(
    os.environ, {"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1", "GROUP_RANK": "0"}
)
@mock.patch("torch.cuda.device_count", return_value=0)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_strategy_choice_ddp_cpu_te(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.local_rank == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy="ddp_spawn", num_processes=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1)
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
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_strategy_choice_ddp_kubeflow(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, KubeflowEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 0
            assert trainer.training_type_plugin.local_rank == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy="ddp", gpus=1, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


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
@mock.patch("torch.cuda.device_count", return_value=0)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
def test_strategy_choice_ddp_cpu_kubeflow(device_count_mock, setup_distributed_mock):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, KubeflowEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 0
            assert trainer.training_type_plugin.local_rank == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy="ddp_spawn", num_processes=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


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
@mock.patch("torch.cuda.device_count", return_value=0)
@mock.patch("pytorch_lightning.plugins.DDPPlugin.setup_distributed", autospec=True)
@pytest.mark.parametrize("strategy", ["ddp", DDPPlugin()])
def test_strategy_choice_ddp_cpu_slurm(device_count_mock, setup_distributed_mock, strategy):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer._accelerator_connector._is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.local_rank == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(fast_dev_run=True, strategy=strategy, num_processes=2, callbacks=[CB()])

    with pytest.raises(SystemExit):
        trainer.fit(model)


def test_unsupported_tpu_choice(monkeypatch):
    import pytorch_lightning.utilities.imports as imports
    from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

    monkeypatch.setattr(imports, "_XLA_AVAILABLE", True)
    monkeypatch.setattr(AcceleratorConnector, "has_tpu", True)
    with pytest.raises(MisconfigurationException, match=r"accelerator='tpu', precision=64\)` is not implemented"):
        Trainer(accelerator="tpu", precision=64)

    with pytest.warns(UserWarning, match=r"accelerator='tpu', precision=16\)` but native AMP is not supported"):
        Trainer(accelerator="tpu", precision=16)
    with pytest.warns(UserWarning, match=r"accelerator='tpu', precision=16\)` but apex AMP is not supported"):
        Trainer(accelerator="tpu", precision=16, amp_backend="apex")


def test_unsupported_ipu_choice(monkeypatch):
    import pytorch_lightning.plugins.training_type.ipu as ipu
    import pytorch_lightning.utilities.imports as imports
    from pytorch_lightning.trainer.connectors.accelerator_connector import AcceleratorConnector

    monkeypatch.setattr(imports, "_IPU_AVAILABLE", True)
    monkeypatch.setattr(ipu, "_IPU_AVAILABLE", True)
    monkeypatch.setattr(AcceleratorConnector, "has_ipu", True)
    with pytest.raises(MisconfigurationException, match=r"accelerator='ipu', precision='bf16'\)` is not supported"):
        Trainer(accelerator="ipu", precision="bf16")
    with pytest.raises(MisconfigurationException, match=r"accelerator='ipu', precision=64\)` is not supported"):
        Trainer(accelerator="ipu", precision=64)


@mock.patch("torch.cuda.is_available", return_value=False)
@mock.patch("pytorch_lightning.utilities.imports._TPU_AVAILABLE", return_value=False)
@mock.patch("pytorch_lightning.utilities.imports._IPU_AVAILABLE", return_value=False)
def test_devices_auto_choice_cpu(is_ipu_available_mock, is_tpu_available_mock, is_gpu_available_mock):
    trainer = Trainer(accelerator="auto", devices="auto")
    assert trainer.devices == 1
    assert trainer.num_processes == 1


@mock.patch("torch.cuda.is_available", return_value=True)
@mock.patch("torch.cuda.device_count", return_value=2)
def test_devices_auto_choice_gpu(is_gpu_available_mock, device_count_mock):
    trainer = Trainer(accelerator="auto", devices="auto")
    assert trainer.devices == 2
    assert trainer.gpus == 2
