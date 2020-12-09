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
from unittest import mock

import pytest

from pytorch_lightning import Trainer, accelerators
from pytorch_lightning.accelerators import Accelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.cluster_environments import ClusterEnvironment, SLURMEnvironment, TorchElasticEnvironment
from tests.base.boring_model import BoringModel


def test_accelerator_choice_cpu(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.CPUAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, TorchElasticEnvironment)

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        callbacks=[CB()]
    )
    trainer.fit(model)


def test_accelerator_choice_ddp_cpu(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator_backend, accelerators.DDPCPUSpawnAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, TorchElasticEnvironment)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_cpu',
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator_backend, accelerators.DDPAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, TorchElasticEnvironment)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp',
        gpus=1,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp_spawn(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator_backend, accelerators.DDPSpawnAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, TorchElasticEnvironment)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_spawn',
        gpus=1,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "SLURM_NTASKS": "2",
    "SLURM_JOB_NAME": "SOME_NAME",
    "SLURM_NODEID": "0",
    "SLURM_LOCALID": "10"
})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp_slurm(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator_backend, accelerators.DDPHPCAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, SLURMEnvironment)
            assert trainer.accelerator_backend.task_idx == 10
            assert trainer.accelerator_backend.cluster_environment.local_rank() == trainer.accelerator_backend.task_idx
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


@mock.patch.dict(os.environ, {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "SLURM_NTASKS": "2",
    "SLURM_JOB_NAME": "SOME_NAME",
    "SLURM_NODEID": "0",
    "LOCAL_RANK": "0",
    "SLURM_LOCALID": "10"
})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp2_slurm(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp2
            assert isinstance(trainer.accelerator_backend, accelerators.DDP2Accelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, SLURMEnvironment)
            assert trainer.accelerator_backend.task_idx == 10
            assert trainer.accelerator_backend.cluster_environment.local_rank() == trainer.accelerator_backend.task_idx

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


@mock.patch.dict(os.environ, {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "WORLD_SIZE": "2",
    "LOCAL_RANK": "10",
    "NODE_RANK": "0"
})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp_te(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator_backend, accelerators.DDPHPCAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, TorchElasticEnvironment)
            assert trainer.accelerator_backend.task_idx == 10
            assert trainer.accelerator_backend.cluster_environment.local_rank() == trainer.accelerator_backend.task_idx
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


@mock.patch.dict(os.environ, {
    "CUDA_VISIBLE_DEVICES": "0,1",
    "WORLD_SIZE": "2",
    "LOCAL_RANK": "10",
    "NODE_RANK": "0"
})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp2_te(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp2
            assert isinstance(trainer.accelerator_backend, accelerators.DDP2Accelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, TorchElasticEnvironment)
            assert trainer.accelerator_backend.task_idx == 10
            assert trainer.accelerator_backend.cluster_environment.local_rank() == trainer.accelerator_backend.task_idx
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


@mock.patch.dict(os.environ, {
    "WORLD_SIZE": "1",
    "LOCAL_RANK": "10",
    "NODE_RANK": "0"
})
@mock.patch('torch.cuda.device_count', return_value=0)
def test_accelerator_choice_ddp_cpu_te(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator_backend, accelerators.DDPCPUHPCAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, TorchElasticEnvironment)
            assert trainer.accelerator_backend.task_idx == 10
            assert trainer.accelerator_backend.cluster_environment.local_rank() == trainer.accelerator_backend.task_idx

            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_cpu',
        num_processes=1,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {
    "SLURM_NTASKS": "1",
    "SLURM_JOB_NAME": "SOME_NAME",
    "SLURM_NODEID": "0",
    "LOCAL_RANK": "0",
    "SLURM_LOCALID": "0"
})
@mock.patch('torch.cuda.device_count', return_value=0)
def test_accelerator_choice_ddp_cpu_slurm(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator_backend, accelerators.DDPCPUHPCAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, SLURMEnvironment)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_cpu',
        num_processes=1,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {
    "SLURM_NTASKS": "1",
    "SLURM_JOB_NAME": "SOME_NAME",
    "SLURM_NODEID": "0",
    "LOCAL_RANK": "0",
    "SLURM_LOCALID": "0"
})
@mock.patch('torch.cuda.device_count', return_value=0)
def test_accelerator_choice_ddp_cpu_custom_cluster(tmpdir):
    """
    Test that we choose the custom cluster even when SLURM or TE flags are around
    """

    class CustomCluster(ClusterEnvironment):
        def master_address(self):
            return 'asdf'

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert trainer.use_ddp
            assert isinstance(trainer.accelerator_backend, accelerators.DDPCPUHPCAccelerator)
            assert isinstance(trainer.accelerator_backend.cluster_environment, CustomCluster)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        plugins=[CustomCluster()],
        fast_dev_run=True,
        accelerator='ddp_cpu',
        num_processes=1,
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {
    "SLURM_NTASKS": "1",
    "SLURM_JOB_NAME": "SOME_NAME",
    "SLURM_NODEID": "0",
    "LOCAL_RANK": "0",
    "SLURM_LOCALID": "0"
})
@mock.patch('torch.cuda.device_count', return_value=0)
def test_custom_accelerator(tmpdir):
    class Accel(Accelerator):
        def init_ddp_connection(
                self,
                global_rank: int,
                world_size: int,
                is_slurm_managing_tasks: bool = True) -> None:
            pass

    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, Accel)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator=Accel(),
        num_processes=1,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {
    "SLURM_NTASKS": "1",
    "SLURM_JOB_NAME": "SOME_NAME",
    "SLURM_NODEID": "0",
    "LOCAL_RANK": "0",
    "SLURM_LOCALID": "0"
})
@mock.patch('torch.cuda.device_count', return_value=0)
def test_dist_backend_accelerator_mapping(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDPCPUHPCAccelerator)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        accelerator='ddp_cpu',
        num_processes=1,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)
