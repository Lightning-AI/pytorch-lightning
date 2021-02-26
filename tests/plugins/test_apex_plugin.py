import os
from unittest import mock

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import ApexMixedPrecisionPlugin
from pytorch_lightning.utilities import _APEX_AVAILABLE


@pytest.mark.skipif(not _APEX_AVAILABLE, reason="test requires apex")
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
def test_amp_choice_default_ddp(mocked_device_count, ddp_backend, gpus):

    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend='apex',
        gpus=gpus,
        accelerator=ddp_backend,
    )
    assert isinstance(trainer.precision_plugin, ApexMixedPrecisionPlugin)


@pytest.mark.skipif(not _APEX_AVAILABLE, reason="test requires apex")
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
def test_amp_choice_custom_ddp(mocked_device_count, ddp_backend, gpus):

    class MyApexPlugin(ApexMixedPrecisionPlugin):
        pass

    trainer = Trainer(
        fast_dev_run=True,
        precision=16,
        amp_backend='apex',
        gpus=gpus,
        accelerator=ddp_backend,
        plugins=[MyApexPlugin(amp_level="O2")],
    )
    assert isinstance(trainer.precision_plugin, MyApexPlugin)
