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
import os
from multiprocessing import Process, Queue

import pytest
from torch.utils.data import DataLoader

import tests.base.develop_pipelines as tpipes
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.accelerators.accelerator import BackendType
from pytorch_lightning.accelerators import TPUAccelerator
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils
from tests.base import EvalModelTemplate
from tests.base.datasets import TrialMNIST
from tests.base.develop_utils import pl_multi_process_test

TPU_AVAILABLE = XLADeviceUtils.tpu_device_exists()

if TPU_AVAILABLE:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    SERIAL_EXEC = xmp.MpSerialExecutor()


_LARGER_DATASET = TrialMNIST(download=True, num_samples=2000, digits=(0, 1, 2, 5, 8))


# 8 cores needs a big dataset
def _serial_train_loader():
    return DataLoader(_LARGER_DATASET, batch_size=32)


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_model_tpu_cores_1(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@pytest.mark.parametrize('tpu_core', [1, 5])
@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_model_tpu_index(tmpdir, tpu_core):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=[tpu_core],
        limit_train_batches=0.4,
        limit_val_batches=0.4,
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)
    assert torch_xla._XLAC._xla_get_default_device() == f'xla:{tpu_core}'


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_model_tpu_cores_8(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=8,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
    )

    model = EvalModelTemplate()
    # 8 cores needs a big dataset
    model.train_dataloader = _serial_train_loader
    model.val_dataloader = _serial_train_loader

    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_model_16bit_tpu_cores_1(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)
    assert os.environ.get('XLA_USE_BF16') == str(1), "XLA_USE_BF16 was not set in environment variables"


@pytest.mark.parametrize('tpu_core', [1, 5])
@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_model_16bit_tpu_index(tmpdir, tpu_core):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=[tpu_core],
        limit_train_batches=0.4,
        limit_val_batches=0.2,
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)
    assert torch_xla._XLAC._xla_get_default_device() == f'xla:{tpu_core}'
    assert os.environ.get('XLA_USE_BF16') == str(1), "XLA_USE_BF16 was not set in environment variables"


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_model_16bit_tpu_cores_8(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=8,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
    )

    model = EvalModelTemplate()
    # 8 cores needs a big dataset
    model.train_dataloader = _serial_train_loader
    model.val_dataloader = _serial_train_loader

    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_model_tpu_early_stop(tmpdir):
    """Test if single TPU core training works"""
    model = EvalModelTemplate()
    trainer = Trainer(
        callbacks=[EarlyStopping()],
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        tpu_cores=1,
    )
    trainer.fit(model)


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_tpu_grad_norm(tmpdir):
    """Test if grad_norm works on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4,
        gradient_clip_val=0.1,
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_dataloaders_passed_to_fit(tmpdir):
    """Test if dataloaders passed to trainer works on TPU"""

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        tpu_cores=8
    )
    result = trainer.fit(model, train_dataloader=model.train_dataloader(), val_dataloaders=model.val_dataloader())
    assert result, "TPU doesn't work with dataloaders passed to fit()."


@pytest.mark.parametrize(
    ['tpu_cores', 'expected_tpu_id'],
    [pytest.param(1, None), pytest.param(8, None), pytest.param([1], 1), pytest.param([8], 8)],
)
def test_tpu_id_to_be_as_expected(tpu_cores, expected_tpu_id):
    """Test if trainer.tpu_id is set as expected"""
    assert Trainer(tpu_cores=tpu_cores).tpu_id == expected_tpu_id


def test_tpu_misconfiguration():
    """Test if trainer.tpu_id is set as expected"""
    with pytest.raises(MisconfigurationException, match="`tpu_cores` can only be"):
        Trainer(tpu_cores=[1, 8])


@pytest.mark.skipif(TPU_AVAILABLE, reason="test requires missing TPU")
def test_exception_when_no_tpu_found(tmpdir):
    """Test if exception is thrown when xla devices are not available"""
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=0.4,
        limit_val_batches=0.2,
        tpu_cores=8,
    )

    with pytest.raises(MisconfigurationException, match='PyTorch XLA not installed.'):
        trainer.fit(model)


@pytest.mark.parametrize('tpu_cores', [1, 8, [1]])
def test_distributed_backend_set_when_using_tpu(tmpdir, tpu_cores):
    """Test if distributed_backend is set to `tpu` when tpu_cores is not None"""
    assert Trainer(tpu_cores=tpu_cores).distributed_backend == "tpu"


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_result_obj_on_tpu(tmpdir):
    seed_everything(1234)
    os.environ['PL_DEV_DEBUG'] = '1'

    batches = 5
    epochs = 2

    model = EvalModelTemplate()
    model.training_step = model.training_step_result_obj
    model.training_step_end = None
    model.training_epoch_end = None
    model.validation_step = model.validation_step_result_obj
    model.validation_step_end = None
    model.validation_epoch_end = None
    model.test_step = model.test_step_result_obj
    model.test_step_end = None
    model.test_epoch_end = None

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        callbacks=[EarlyStopping()],
        log_every_n_steps=2,
        limit_train_batches=batches,
        weights_summary=None,
        tpu_cores=8
    )

    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
@pl_multi_process_test
def test_broadcast_on_tpu():
    """ Checks if an object from the master process is broadcasted to other processes correctly"""
    def test_broadcast(rank):
        trainer = Trainer(tpu_cores=8)
        backend = TPUAccelerator(trainer)
        obj = ("ver_0.5", "logger_name", rank)
        result = backend.broadcast(obj)
        assert result == ("ver_0.5", "logger_name", 0)

    xmp.spawn(test_broadcast, nprocs=8, start_method='fork')
