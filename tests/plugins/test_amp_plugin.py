import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
from pytorch_lightning.utilities import _NATIVE_AMP_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel


@pytest.mark.skipif(not _NATIVE_AMP_AVAILABLE, reason="Minimal PT version is set to 1.6")
@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0"
    }
)
@mock.patch('torch.cuda.device_count', return_value=2)
@pytest.mark.parametrize(
    ['ddp_backend', 'gpus', 'num_processes'],
    [('ddp_cpu', None, 2), ('ddp', 2, 0), ('ddp2', 2, 0), ('ddp_spawn', 2, 0)],
)
def on_fit_start(tmpdir, ddp_backend, gpus, num_processes):

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.precision_plugin, NativeMixedPrecisionPlugin)
            raise SystemExit()

    def train():
        model = BoringModel()
        trainer = Trainer(
            fast_dev_run=True,
            precision=16,
            amp_backend='native',
            gpus=gpus,
            num_processes=num_processes,
            accelerator=ddp_backend,
            callbacks=[CB()],
        )
        trainer.fit(model)

    if ddp_backend == "ddp_cpu":
        with pytest.raises(MisconfigurationException, match="MP is only available on GPU"):
            train()
    else:
        with pytest.raises(SystemExit):
            train()


@pytest.mark.skipif(not _NATIVE_AMP_AVAILABLE, reason="Minimal PT version is set to 1.6")
@mock.patch.dict(
    os.environ, {
        "CUDA_VISIBLE_DEVICES": "0,1",
        "SLURM_NTASKS": "2",
        "SLURM_JOB_NAME": "SOME_NAME",
        "SLURM_NODEID": "0",
        "LOCAL_RANK": "0",
        "SLURM_LOCALID": "0"
    }
)
@mock.patch('torch.cuda.device_count', return_value=2)
@pytest.mark.parametrize(
    ['ddp_backend', 'gpus', 'num_processes'],
    [('ddp_cpu', None, 2), ('ddp', 2, 0), ('ddp2', 2, 0), ('ddp_spawn', 2, 0)],
)
def test_amp_choice_custom_ddp_cpu(tmpdir, ddp_backend, gpus, num_processes):

    class MyNativeAMP(NativeMixedPrecisionPlugin):
        pass

    class CB(Callback):

        def on_fit_start(self, trainer, pl_module):
            assert isinstance(trainer.precision_plugin, MyNativeAMP)
            raise SystemExit()

    model = BoringModel()
    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend='native',
        num_processes=num_processes,
        accelerator=ddp_backend,
        plugins=[MyNativeAMP()],
        callbacks=[CB()],
    )

    with pytest.raises(SystemExit):
        trainer.fit(model)


class GradientUnscaleBoringModel(BoringModel):

    def on_after_backward(self):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
        if not (torch.isinf(norm) or torch.isnan(norm)):
            assert norm.item() < 15.


@pytest.mark.skipif(not _NATIVE_AMP_AVAILABLE, reason="Minimal PT version is set to 1.6")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_amp_gradient_unscale(tmpdir):
    model = GradientUnscaleBoringModel()

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=os.getcwd(),
        limit_train_batches=2,
        limit_test_batches=2,
        limit_val_batches=2,
        amp_backend='native',
        accelerator='ddp_spawn',
        gpus=2,
        precision=16,
        track_grad_norm=2,
        log_every_n_steps=1,
    )
    trainer.fit(model)


class UnscaleAccumulateGradBatchesBoringModel(BoringModel):

    def on_after_backward(self):
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), 2)
        if not (torch.isinf(norm) or torch.isnan(norm)):
            assert norm.item() < 15.


@pytest.mark.skipif(not _NATIVE_AMP_AVAILABLE, reason="Minimal PT version is set to 1.6")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_amp_gradient_unscale_accumulate_grad_batches(tmpdir):
    model = UnscaleAccumulateGradBatchesBoringModel()

    trainer = Trainer(
        max_epochs=2,
        default_root_dir=os.getcwd(),
        limit_train_batches=2,
        limit_test_batches=2,
        limit_val_batches=2,
        amp_backend='native',
        accelerator='ddp_spawn',
        gpus=2,
        precision=16,
        track_grad_norm=2,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
    )
    trainer.fit(model)
