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

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.gpu import GPUAccelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import (
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
from pytorch_lightning.plugins.environments import LightningEnvironment, SLURMEnvironment, TorchElasticEnvironment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf


def test_accelerator_choice_cpu(tmpdir):
    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
    )
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.training_type_plugin, SingleDevicePlugin)


def test_accelerator_choice_ddp_cpu(tmpdir):
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_cpu',
    )
    assert isinstance(trainer.accelerator, CPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPSpawnPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch('torch.cuda.device_count', return_value=2)
@mock.patch('torch.cuda.is_available', return_value=True)
def test_accelerator_choice_ddp(cuda_available_mock, device_count_mock):
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp',
        gpus=1,
    )
    assert isinstance(trainer.accelerator, GPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch('torch.cuda.device_count', return_value=2)
@mock.patch('torch.cuda.is_available', return_value=True)
def test_accelerator_choice_ddp_spawn(cuda_available_mock, device_count_mock):
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_spawn',
        gpus=1,
    )
    assert isinstance(trainer.accelerator, GPUAccelerator)
    assert isinstance(trainer.training_type_plugin, DDPSpawnPlugin)
    assert isinstance(trainer.training_type_plugin.cluster_environment, LightningEnvironment)


@RunIf(min_gpus=2)
@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "SLURM_PROCID": "1",
        "SLURM_LOCALID": "1",
    }
)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_accelerator_choice_ddp_slurm(setup_distributed_mock):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert trainer.accelerator_connector.is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.task_idx == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp',
        gpus=2,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=2)
@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "SLURM_PROCID": "1",
        "SLURM_LOCALID": "1"
    }
)
@mock.patch('torch.cuda.device_count', return_value=2)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_accelerator_choice_ddp2_slurm(device_count_mock, setup_distributed_mock):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp2
            assert trainer.accelerator_connector.is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDP2Plugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.task_idx == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp2',
        gpus=2,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1)
@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
    }
)
@mock.patch('torch.cuda.device_count', return_value=2)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_accelerator_choice_ddp_te(device_count_mock, setup_distributed_mock):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.task_idx == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp',
        gpus=2,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@RunIf(min_gpus=1)
@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
    }
)
@mock.patch('torch.cuda.device_count', return_value=2)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_accelerator_choice_ddp2_te(device_count_mock, setup_distributed_mock):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp2
            assert isinstance(trainer.accelerator, GPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDP2Plugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.task_idx == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp2',
        gpus=2,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(
    os.environ, {
        "WORLD_SIZE": "2",
        "LOCAL_WORLD_SIZE": "2",
        "RANK": "1",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "0",
    }
)
@mock.patch('torch.cuda.device_count', return_value=0)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_accelerator_choice_ddp_cpu_te(device_count_mock, setup_distributed_mock):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, TorchElasticEnvironment)
            assert trainer.training_type_plugin.cluster_environment.local_rank() == 1
            assert trainer.training_type_plugin.task_idx == 1
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_cpu',
        num_processes=2,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(
    os.environ, {
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    }
)
@mock.patch('torch.cuda.device_count', return_value=0)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_accelerator_choice_ddp_cpu_slurm(device_count_mock, setup_distributed_mock):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert trainer.accelerator_connector.is_slurm_managing_tasks
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, SLURMEnvironment)
            assert trainer.training_type_plugin.task_idx == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_cpu',
        num_processes=2,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(
    os.environ, {
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    }
)
@mock.patch('torch.cuda.device_count', return_value=0)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_accelerator_choice_ddp_cpu_custom_cluster(device_count_mock, setup_distributed_mock):
    """
    Test that we choose the custom cluster even when SLURM or TE flags are around
    """

    class CustomCluster(LightningEnvironment):

        def master_address(self):
            return 'asdf'

        def creates_children(self) -> bool:
            return True

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert isinstance(trainer.training_type_plugin.cluster_environment, CustomCluster)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        plugins=[CustomCluster()],
        fast_dev_run=True,
        accelerator='ddp_cpu',
        num_processes=2,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(
    os.environ, {
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0"
    }
)
@mock.patch('torch.cuda.device_count', return_value=0)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_custom_accelerator(device_count_mock, setup_distributed_mock):

    class Accel(Accelerator):
        pass

    class Prec(PrecisionPlugin):
        pass

    class TrainTypePlugin(SingleDevicePlugin):
        pass

    accelerator = Accel(
        training_type_plugin=TrainTypePlugin(device=torch.device("cpu")),
        precision_plugin=Prec(),
    )
    trainer = Trainer(
        accelerator=accelerator,
        fast_dev_run=True,
        num_processes=2,
    )
    assert isinstance(trainer.accelerator, Accel)
    assert isinstance(trainer.training_type_plugin, TrainTypePlugin)
    assert isinstance(trainer.precision_plugin, Prec)


@mock.patch.dict(
    os.environ, {
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_PROCID": "0",
        "SLURM_LOCALID": "0",
    }
)
@mock.patch('torch.cuda.device_count', return_value=0)
@mock.patch('pytorch_lightning.plugins.DDPPlugin.setup_distributed', autospec=True)
def test_dist_backend_accelerator_mapping(device_count_mock, setup_distributed_mock):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator, CPUAccelerator)
            assert isinstance(trainer.training_type_plugin, DDPPlugin)
            assert trainer.training_type_plugin.task_idx == 0
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_cpu',
        num_processes=2,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch("pytorch_lightning.utilities._IS_INTERACTIVE", return_value=True)
@mock.patch('torch.cuda.device_count', return_value=2)
def test_ipython_incompatible_backend_error(*_):
    with pytest.raises(MisconfigurationException, match="backend ddp is not compatible"):
        Trainer(accelerator="ddp", gpus=2)

    with pytest.raises(MisconfigurationException, match="backend ddp is not compatible"):
        Trainer(accelerator="ddp_cpu", num_processes=2)

    with pytest.raises(MisconfigurationException, match="backend ddp2 is not compatible"):
        Trainer(accelerator="ddp2", gpus=2)


@pytest.mark.parametrize(
    ["accelerator", "plugin"],
    [('ddp_spawn', 'ddp_sharded'), (None, 'ddp_sharded')],
)
def test_plugin_accelerator_choice(accelerator: Optional[str], plugin: str):
    """Ensure that when a plugin and accelerator is passed in, that the plugin takes precedent."""
    trainer = Trainer(accelerator=accelerator, plugins=plugin, num_processes=2)
    assert isinstance(trainer.accelerator.training_type_plugin, DDPShardedPlugin)

    trainer = Trainer(plugins=plugin, num_processes=2)
    assert isinstance(trainer.accelerator.training_type_plugin, DDPShardedPlugin)


@pytest.mark.parametrize(["accelerator", "plugin"], [
    ('ddp', DDPPlugin),
    ('ddp_spawn', DDPSpawnPlugin),
    ('ddp_sharded', DDPShardedPlugin),
    ('ddp_sharded_spawn', DDPSpawnShardedPlugin),
    pytest.param('deepspeed', DeepSpeedPlugin, marks=RunIf(deepspeed=True)),
])
@mock.patch('torch.cuda.is_available', return_value=True)
@mock.patch('torch.cuda.device_count', return_value=2)
@pytest.mark.parametrize("gpus", [1, 2])
def test_accelerator_choice_multi_node_gpu(
    mock_is_available, mock_device_count, tmpdir, accelerator: str, plugin: ParallelPlugin, gpus: int
):
    trainer = Trainer(
        accelerator=accelerator,
        default_root_dir=tmpdir,
        num_nodes=2,
        gpus=gpus,
    )
    assert isinstance(trainer.training_type_plugin, plugin)
