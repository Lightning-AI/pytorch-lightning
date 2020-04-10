import os

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import (
    LightningTestModel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_amp_single_gpu(tmpdir):
    """Make sure DDP + AMP work."""
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=1,
        distributed_backend='ddp',
        precision=16
    )

    tutils.run_model_test(trainer_options, model)


@pytest.mark.spawn
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_no_amp_single_gpu(tmpdir):
    """Make sure DDP + AMP work."""
    tutils.reset_seed()

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=1,
        distributed_backend='dp',
        precision=16
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_amp_gpu_ddp(tmpdir):
    """Make sure DDP + AMP work."""
    tutils.reset_seed()
    tutils.set_random_master_port()

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=2,
        distributed_backend='ddp',
        precision=16
    )

    tutils.run_model_test(trainer_options, model)


@pytest.mark.spawn
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_amp_gpu_ddp_slurm_managed(tmpdir):
    """Make sure DDP + AMP work."""
    tutils.reset_seed()

    # simulate setting slurm flags
    tutils.set_random_master_port()
    os.environ['SLURM_LOCALID'] = str(0)

    hparams = tutils.get_default_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        max_epochs=1,
        gpus=[0],
        distributed_backend='ddp',
        precision=16
    )

    # exp file to get meta
    logger = tutils.get_default_testtube_logger(tmpdir, False)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # add these to the trainer options
    trainer_options['checkpoint_callback'] = checkpoint
    trainer_options['logger'] = logger

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.is_slurm_managing_tasks = True
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test root model address
    assert trainer.resolve_root_node_address('abc') == 'abc'
    assert trainer.resolve_root_node_address('abc[23]') == 'abc23'
    assert trainer.resolve_root_node_address('abc[23-24]') == 'abc23'
    assert trainer.resolve_root_node_address('abc[23-24, 45-40, 40]') == 'abc23'


def test_cpu_model_with_amp(tmpdir):
    """Make sure model trains on CPU."""
    tutils.reset_seed()

    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        logger=tutils.get_default_testtube_logger(tmpdir),
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        precision=16
    )

    model, hparams = tutils.get_default_model()

    with pytest.raises((MisconfigurationException, ModuleNotFoundError)):
        tutils.run_model_test(trainer_options, model, on_gpu=False)


@pytest.mark.spawn
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_amp_gpu_dp(tmpdir):
    """Make sure DP + AMP work."""
    tutils.reset_seed()

    model, hparams = tutils.get_default_model()
    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus='0, 1',  # test init with gpu string
        distributed_backend='dp',
        precision=16
    )

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1


if __name__ == '__main__':
    pytest.main([__file__])
