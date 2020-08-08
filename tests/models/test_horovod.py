import json
import os
import platform
import shlex
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch

import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from tests.base import EvalModelTemplate
from tests.base.models import TestGAN

try:
    from horovod.common.util import nccl_built
except ImportError:
    HOROVOD_AVAILABLE = False
else:
    HOROVOD_AVAILABLE = True


# This script will run the actual test model training in parallel
TEST_SCRIPT = os.path.join(os.path.dirname(__file__), 'data', 'horovod', 'train_default_model.py')


def _nccl_available():
    if not HOROVOD_AVAILABLE:
        return False

    try:
        return nccl_built()
    except AttributeError:
        # Horovod 0.19.1 nccl_built() does not yet work with Python 3.8:
        # See: https://github.com/horovod/horovod/issues/1891
        return False


def _run_horovod(trainer_options, on_gpu=False):
    """Execute the training script across multiple workers in parallel."""
    num_processes = trainer_options.get('gpus', 2)
    # gpus trainer argument does not apply for horovod
    trainer_options.update(gpus=None)
    tutils.reset_seed()
    cmdline = [
        'horovodrun',
        '-np', str(num_processes),
        sys.executable, TEST_SCRIPT,
        '--trainer-options', shlex.quote(json.dumps(trainer_options))
    ]
    if on_gpu:
        cmdline += ['--on-gpu']
    exit_code = subprocess.call(' '.join(cmdline), shell=True, env=os.environ.copy())
    assert exit_code == 0


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_cpu(tmpdir):
    """Test Horovod running multi-process on CPU."""
    trainer_options = dict(
        default_root_dir=str(tmpdir),
        weights_save_path=str(tmpdir),
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        distributed_backend='horovod',
        deterministic=True,
    )
    _run_horovod(trainer_options)


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_cpu_implicit(tmpdir):
    """Test Horovod without specifying a backend, inferring from env set by `horovodrun`."""
    trainer_options = dict(
        default_root_dir=str(tmpdir),
        weights_save_path=str(tmpdir),
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        deterministic=True,
    )
    _run_horovod(trainer_options)


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
@pytest.mark.skipif(not _nccl_available(), reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_horovod_multi_gpu(tmpdir):
    """Test Horovod with multi-GPU support."""
    trainer_options = dict(
        default_root_dir=str(tmpdir),
        weights_save_path=str(tmpdir),
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        gpus=2,
        deterministic=True,
        distributed_backend='horovod'
    )
    _run_horovod(trainer_options, on_gpu=True)


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
@pytest.mark.skipif(not _nccl_available(), reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_horovod_transfer_batch_to_gpu(tmpdir):
    class TestTrainingStepModel(EvalModelTemplate):
        def training_step(self, batch, *args, **kwargs):
            x, y = batch
            assert str(x.device) != 'cpu'
            assert str(y.device) != 'cpu'
            return super(TestTrainingStepModel, self).training_step(batch, *args, **kwargs)

        def validation_step(self, batch, *args, **kwargs):
            x, y = batch
            assert str(x.device) != 'cpu'
            assert str(y.device) != 'cpu'
            return super(TestTrainingStepModel, self).validation_step(batch, *args, **kwargs)

    hparams = EvalModelTemplate.get_default_hparams()
    model = TestTrainingStepModel(**hparams)

    trainer_options = dict(
        default_root_dir=str(tmpdir),
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        gpus=1,
        deterministic=True,
        distributed_backend='horovod'
    )
    tpipes.run_model_test_without_loggers(trainer_options, model)


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_multi_optimizer(tmpdir):
    model = TestGAN(**EvalModelTemplate.get_default_hparams())

    # fit model
    trainer = Trainer(
        default_root_dir=str(tmpdir),
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        deterministic=True,
        distributed_backend='horovod',
    )
    result = trainer.fit(model)
    assert result == 1, 'model failed to complete'

    assert len(trainer.optimizers) == 2
    for i, optimizer in enumerate(trainer.optimizers):
        assert hasattr(optimizer, 'synchronize'), 'optimizer has not been wrapped into DistributedOptimizer'

    def get_model_params(model):
        return set([p for p in model.parameters()])

    def get_optimizer_params(optimizer):
        return set([p for group in optimizer.param_groups for p in group.get('params', [])])

    assert get_model_params(model.generator) != get_model_params(model.discriminator)
    assert get_model_params(model.generator) == get_optimizer_params(trainer.optimizers[0])
    assert get_model_params(model.discriminator) == get_optimizer_params(trainer.optimizers[1])


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_multi_optimizer_with_scheduling_stepping(tmpdir):
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)
    model.configure_optimizers = model.configure_optimizers__multiple_schedulers

    num_workers = 8
    init_lr = hparams.get('learning_rate') * num_workers

    with patch('pytorch_lightning.trainer.distrib_parts.hvd.size') as mock_hvd_size:
        mock_hvd_size.return_value = 8

        # fit model
        trainer = Trainer(
            default_root_dir=tmpdir,
            max_epochs=1,
            limit_val_batches=0.5,
            limit_train_batches=0.2,
            distributed_backend='horovod'
        )
        results = trainer.fit(model)
        assert results == 1

    adjusted_lr1 = [pg['lr'] for pg in trainer.optimizers[0].param_groups][0]
    adjusted_lr2 = [pg['lr'] for pg in trainer.optimizers[1].param_groups][0]

    # Called ones after end of epoch with gamma=0.1
    assert pytest.approx(init_lr * 0.1) == adjusted_lr1

    # Called every 3 steps, meaning for 1 epoch of 11 batches, it is called 3 times with gamma=0.1
    assert pytest.approx(init_lr * 0.1) == adjusted_lr2
