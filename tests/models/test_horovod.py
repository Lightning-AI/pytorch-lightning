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
# limitations under the License.
import json
import os
import platform
import shlex
import subprocess
import sys

import numpy as np
import pytest
from sklearn.metrics import accuracy_score
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.horovod_accelerator import HorovodAccelerator
from pytorch_lightning.core.step_result import EvalResult, Result, TrainResult
from pytorch_lightning.metrics.classification.accuracy import Accuracy
from pytorch_lightning.utilities import _module_available, APEX_AVAILABLE, HOROVOD_AVAILABLE, NATIVE_AMP_AVAILABLE
from tests.base import EvalModelTemplate
from tests.base.boring_model import BoringModel
import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from tests.base.models import BasicGAN

if HOROVOD_AVAILABLE:
    import horovod
    import horovod.torch as hvd

# This script will run the actual test model training in parallel
TEST_SCRIPT = os.path.join(os.path.dirname(__file__), 'data', 'horovod', 'train_default_model.py')

try:
    from horovod.common.util import nccl_built
    nccl_built()
except (ImportError, ModuleNotFoundError, AttributeError):
    HOROVOD_NCCL_AVAILABLE = False
finally:
    HOROVOD_NCCL_AVAILABLE = True


def _run_horovod(trainer_options, on_gpu=False):
    """Execute the training script across multiple workers in parallel."""
    num_processes = trainer_options.get('gpus', 2)
    # for Horovod, we interpret `gpus` to be set per worker
    trainer_options.update(gpus=1 if on_gpu else None)
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
        accelerator='horovod',
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
@pytest.mark.skipif(not HOROVOD_NCCL_AVAILABLE, reason="test requires Horovod with NCCL support")
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
        accelerator='horovod',
    )
    _run_horovod(trainer_options, on_gpu=True)


@pytest.mark.skip(reason="Horovod has a problem with broadcast when using apex?")
@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
@pytest.mark.skipif(not HOROVOD_NCCL_AVAILABLE, reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not APEX_AVAILABLE, reason="test requires apex")
def test_horovod_apex(tmpdir):
    """Test Horovod with multi-GPU support using apex amp."""
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
        accelerator='horovod',
        amp_backend='apex',
        precision=16,
    )
    _run_horovod(trainer_options, on_gpu=True)


@pytest.mark.skip(reason="Skip till Horovod fixes integration with Native torch.cuda.amp")
@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
@pytest.mark.skipif(not HOROVOD_NCCL_AVAILABLE, reason="test requires Horovod with NCCL support")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
@pytest.mark.skipif(not NATIVE_AMP_AVAILABLE, reason="test requires torch.cuda.amp")
def test_horovod_amp(tmpdir):
    """Test Horovod with multi-GPU support using native amp."""
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
        accelerator='horovod',
        amp_backend='native',
        precision=16,
    )
    _run_horovod(trainer_options, on_gpu=True)


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
@pytest.mark.skipif(not HOROVOD_NCCL_AVAILABLE, reason="test requires Horovod with NCCL support")
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
        accelerator='horovod',
    )
    tpipes.run_model_test_without_loggers(trainer_options, model)


@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_horovod_multi_optimizer(tmpdir):
    model = BasicGAN(**EvalModelTemplate.get_default_hparams())

    # fit model
    trainer = Trainer(
        default_root_dir=str(tmpdir),
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        deterministic=True,
        accelerator='horovod',
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


@pytest.mark.skipif(not HOROVOD_AVAILABLE, reason="Horovod is unavailable")
@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_result_reduce_horovod(tmpdir):
    """Make sure result logging works with Horovod.

    This test mirrors tests/core/test_results.py::_ddp_test_fn
    """
    tutils.reset_seed()
    tutils.set_random_master_port()

    def hvd_test_fn():
        path_here = os.path.abspath(os.path.dirname(__file__))
        path_root = os.path.abspath(os.path.join(path_here, '..', '..'))
        sys.path.insert(0, os.path.abspath(path_root))

        class TestModel(BoringModel):
            def training_step(self, batch, batch_idx):
                self.training_step_called = True

                tensor = torch.tensor([1.0])
                self.log("test_tensor", tensor, sync_dist=True, sync_dist_op='sum',
                         on_step=True, on_epoch=True)

                res = self._results

                # Check that `tensor` is summed across all ranks automatically
                assert res["test_tensor"].item() == hvd.size(), \
                    "Result-Log does not work properly with Horovod and Tensors"

            def training_epoch_end(self, outputs) -> None:
                assert len(outputs) == 0

        model = TestModel()
        model.val_dataloader = None

        trainer = Trainer(
            default_root_dir=tmpdir,
            limit_train_batches=2,
            limit_val_batches=2,
            max_epochs=1,
            log_every_n_steps=1,
            weights_summary=None,
        )

        trainer.fit(model)

    horovod.run(hvd_test_fn, np=2)


@pytest.mark.skipif(not HOROVOD_AVAILABLE, reason="Horovod is unavailable")
@pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
def test_accuracy_metric_horovod():
    num_batches = 10
    batch_size = 16
    threshold = 0.5

    def sk_metric(preds, target):
        sk_preds = (preds.view(-1).numpy() >= threshold).astype(np.uint8)
        sk_target = target.view(-1).numpy()
        return accuracy_score(y_true=sk_target, y_pred=sk_preds)

    preds = torch.rand(num_batches, batch_size)
    target = torch.randint(high=2, size=(num_batches, batch_size))

    def _compute_batch():
        trainer = Trainer(
            fast_dev_run=True,
            accelerator='horovod',
        )

        accelerator_backend = trainer.accelerator_connector.select_accelerator()
        assert isinstance(accelerator_backend, HorovodAccelerator)

        metric = Accuracy(compute_on_step=True,
                          dist_sync_on_step=True,
                          dist_sync_fn=accelerator_backend.gather_all_tensors,
                          threshold=threshold)

        for i in range(hvd.rank(), num_batches, hvd.size()):
            batch_result = metric(preds[i], target[i])
            if hvd.rank() == 0:
                dist_preds = torch.stack([preds[i + r] for r in range(hvd.size())])
                dist_target = torch.stack([target[i + r] for r in range(hvd.size())])
                sk_batch_result = sk_metric(dist_preds, dist_target)
                assert np.allclose(batch_result.numpy(), sk_batch_result)

        # check on all batches on all ranks
        result = metric.compute()
        assert isinstance(result, torch.Tensor)

        total_preds = torch.stack([preds[i] for i in range(num_batches)])
        total_target = torch.stack([target[i] for i in range(num_batches)])
        sk_result = sk_metric(total_preds, total_target)

        assert np.allclose(result.numpy(), sk_result)

    horovod.run(_compute_batch, np=2)

# @pytest.mark.skipif(platform.system() == "Windows", reason="Horovod is not supported on Windows")
# def test_horovod_multi_optimizer_with_scheduling_stepping(tmpdir):
#     hparams = EvalModelTemplate.get_default_hparams()
#     model = EvalModelTemplate(**hparams)
#     model.configure_optimizers = model.configure_optimizers__multiple_schedulers
#
#     num_workers = 8
#     init_lr = hparams.get('learning_rate') * num_workers
#
#     with patch('pytorch_lightning.accelerators.horovod_backend.hvd.size') as mock_hvd_size:
#         mock_hvd_size.return_value = 8
#
#         # fit model
#         trainer = Trainer(
#             default_root_dir=tmpdir,
#             max_epochs=1,
#             limit_val_batches=0.5,
#             limit_train_batches=0.2,
#             distributed_backend='horovod'
#         )
#         results = trainer.fit(model)
#         assert results == 1
#
#     adjusted_lr1 = [pg['lr'] for pg in trainer.optimizers[0].param_groups][0]
#     adjusted_lr2 = [pg['lr'] for pg in trainer.optimizers[1].param_groups][0]
#
#     # Called ones after end of epoch with gamma=0.1
#     assert pytest.approx(init_lr * 0.1) == adjusted_lr1
#
#     # Called every 3 steps, meaning for 1 epoch of 11 batches, it is called 3 times with gamma=0.1
#     assert pytest.approx(init_lr * 0.1) == adjusted_lr2
