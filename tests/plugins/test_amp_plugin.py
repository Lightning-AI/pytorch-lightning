import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin
from pytorch_lightning.utilities import _NATIVE_AMP_AVAILABLE
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
    ['ddp_backend', 'gpus'],
    [('ddp', 2), ('ddp2', 2), ('ddp_spawn', 2)],
)
def test_amp_choice_custom_ddp_cpu(device_count_mock, ddp_backend, gpus):

    class MyNativeAMP(NativeMixedPrecisionPlugin):
        pass

    trainer = Trainer(
        precision=16,
        amp_backend='native',
        accelerator=ddp_backend,
        plugins=[MyNativeAMP()],
    )
    assert isinstance(trainer.precision_plugin, MyNativeAMP)


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
