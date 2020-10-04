import pytest
import torch
import os
from tests.backends import ddp_model
from tests.base.boring_model import BoringModel
from pytorch_lightning.callbacks import Callback
from tests.utilities.dist import call_training_script
from pytorch_lightning import accelerators, Trainer
from unittest import mock


def test_accelerator_choice_cpu(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.CPUBackend)

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        callbacks=[CB()]
    )
    trainer.fit(model)


def test_accelerator_choice_ddp_cpu(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDPCPUSpawnBackend)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        distributed_backend='ddp_cpu',
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDPBackend)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        distributed_backend='ddp',
        gpus=1,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp_spawn(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDPSpawnBackend)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        distributed_backend='ddp_spawn',
        gpus=1,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1",
                              "SLURM_NTASKS": "2",
                              "SLURM_JOB_NAME": "SOME_NAME",
                              "SLURM_NODEID": "0",
                              "SLURM_LOCALID": "0"})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp_slurm(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDPSLURMBackend)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        distributed_backend='ddp',
        gpus=2,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1",
                              "SLURM_NTASKS": "2",
                              "SLURM_JOB_NAME": "SOME_NAME",
                              "SLURM_NODEID": "0",
                              "LOCAL_RANK": "0",
                              "SLURM_LOCALID": "0"})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp2_slurm(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDP2Backend)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        distributed_backend='ddp2',
        gpus=2,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1",
                              "WORLD_SIZE": "2",
                              "LOCAL_RANK": "0",
                              "NODE_RANK": "0"})
@mock.patch('torch.cuda.device_count', return_value=2)
def test_accelerator_choice_ddp_te(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDPTorchElasticBackend)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        distributed_backend='ddp',
        gpus=2,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


@mock.patch.dict(os.environ, {"WORLD_SIZE": "1",
                              "LOCAL_RANK": "0",
                              "NODE_RANK": "0"})
@mock.patch('torch.cuda.device_count', return_value=0)
def test_accelerator_choice_ddp_cpu_te(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDPCPUTorchElasticBackend)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        distributed_backend='ddp_cpu',
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
                              "SLURM_LOCALID": "0"})
@mock.patch('torch.cuda.device_count', return_value=0)
def test_accelerator_choice_ddp_cpu_slurm(tmpdir):
    class CB(Callback):
        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.accelerator_backend, accelerators.DDPCPUSLURMBackend)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        distributed_backend='ddp_cpu',
        num_processes=1,
        callbacks=[CB()]
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)
