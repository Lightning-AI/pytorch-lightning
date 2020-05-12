import json
import os
import platform
import shlex
import subprocess
import sys

import pytest
import torch

import tests.base.utils as tutils
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
    tutils.reset_seed()
    cmdline = [
        'horovodrun',
        '-np', '2',
        sys.executable, TEST_SCRIPT,
        '--trainer-options', shlex.quote(json.dumps(trainer_options))
    ]
    if on_gpu:
        cmdline += ['--on-gpu']
    exit_code = subprocess.call(' '.join(cmdline), shell=True, env=os.environ.copy())
    assert exit_code == 0


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Horovod not yet supported in Python 3.8")
@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_cpu(tmpdir):
    """Test Horovod running multi-process on CPU."""
    trainer_options = dict(
        default_root_dir=str(tmpdir),
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        distributed_backend='horovod',
        deterministic=True,
    )
    _run_horovod(trainer_options)


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Horovod not yet supported in Python 3.8")
@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_cpu_implicit(tmpdir):
    """Test Horovod without specifying a backend, inferring from env set by `horovodrun`."""
    trainer_options = dict(
        default_root_dir=str(tmpdir),
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        deterministic=True,
    )
    _run_horovod(trainer_options)


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Horovod not yet supported in Python 3.8")
@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
@pytest.mark.skipif(not _nccl_available(), reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_horovod_multi_gpu(tmpdir):
    """Test Horovod with multi-GPU support."""
    trainer_options = dict(
        default_root_dir=str(tmpdir),
        gradient_clip_val=1.0,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        gpus=1,
        deterministic=True,
        distributed_backend='horovod'
    )
    _run_horovod(trainer_options, on_gpu=True)


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Horovod not yet supported in Python 3.8")
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
    model = TestTrainingStepModel(hparams)

    trainer_options = dict(
        default_root_dir=str(tmpdir),
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        gpus=1,
        deterministic=True,
        distributed_backend='horovod'
    )
    tutils.run_model_test_without_loggers(trainer_options, model)


@pytest.mark.skipif(sys.version_info >= (3, 8), reason="Horovod not yet supported in Python 3.8")
@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_multi_optimizer(tmpdir):
    hparams = EvalModelTemplate.get_default_hparams()
    model = TestGAN(hparams)

    trainer_options = dict(
        default_root_dir=str(tmpdir),
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        deterministic=True,
        distributed_backend='horovod'
    )

    # fit model
    trainer = Trainer(**trainer_options)
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
