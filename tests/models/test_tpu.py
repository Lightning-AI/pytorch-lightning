import os
from unittest.mock import patch

import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
import tests.base.develop_pipelines as tpipes
from tests.base.datasets import TrialMNIST
from torch.utils.data import DataLoader

try:
    import torch_xla
    # TODO: The tests are aborted if the following lines are uncommented. Must be resolved with XLA team
    # device = torch_xla.core.xla_model.xla_device()
    # device_type = torch_xla.core.xla_model.xla_device_hw(device)
    # TPU_AVAILABLE = device_type == 'TPU'
except ImportError:
    TPU_AVAILABLE = False
else:
    TPU_AVAILABLE = True


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_model_tpu_cores_1(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_model_tpu_idx_1(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=[1],
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)
    assert torch_xla._XLAC._xla_get_default_device() == 'xla:1'


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_model_tpu_idx_8(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=[8],
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)
    assert torch_xla._XLAC._xla_get_default_device() == 'xla:8'


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_model_tpu_cores_8(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=8,
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()

    # 8 cores needs a big dataset
    def long_train_loader():
        dataset = DataLoader(TrialMNIST(download=True, num_samples=15000, digits=(0, 1, 2, 5, 8)), batch_size=32)
        return dataset
    model.train_dataloader = long_train_loader
    model.val_dataloader = long_train_loader

    tpipes.run_model_test(trainer_options, model, on_gpu=False, with_hpc=False)


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_model_16bit_tpu_cores_1(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=1,
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)
    assert os.environ.get('XLA_USE_BF16') == str(1), "XLA_USE_BF16 was not set in environment variables"


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_model_16bit_tpu_idx_1(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=[1],
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model, on_gpu=False)
    assert torch_xla._XLAC._xla_get_default_device() == 'xla:1'
    assert os.environ.get('XLA_USE_BF16') == str(1), "XLA_USE_BF16 was not set in environment variables"


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_model_16bit_tpu_cores_8(tmpdir):
    """Make sure model trains on TPU."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        precision=16,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        tpu_cores=8,
        limit_train_batches=0.4,
        limit_val_batches=0.4
    )

    model = EvalModelTemplate()

    # 8 cores needs a big dataset
    def long_train_loader():
        dataset = DataLoader(TrialMNIST(download=True, num_samples=15000, digits=(0, 1, 2, 5, 8)), batch_size=32)
        return dataset
    model.train_dataloader = long_train_loader
    model.val_dataloader = long_train_loader

    tpipes.run_model_test(trainer_options, model, on_gpu=False)
    assert os.environ.get('XLA_USE_BF16') == str(1), "XLA_USE_BF16 was not set in environment variables"


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_early_stop_checkpoints_on_tpu(tmpdir):
    """Test if single TPU core training works"""
    model = EvalModelTemplate()
    trainer = Trainer(
        early_stop_callback=True,
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        tpu_cores=[8],
    )
    trainer.fit(model)
    assert torch_xla._XLAC._xla_get_default_device() == 'xla:8'


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_model_16bit_tpu_index_1_8(tmpdir):
    """Test if distributed TPU core training works"""
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        tpu_cores=[1, 8],
    )
    trainer.fit(model)
    assert trainer.tpu_id is None


@pytest.mark.skipif(not TPU_AVAILABLE, reason="test requires TPU machine")
def test_dataloaders_passed_to_fit(tmpdir):
    """Test if dataloaders passed to trainer works on TPU"""

    model = EvalModelTemplate()

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        tpu_cores=8,
    )
    result = trainer.fit(
        model,
        train_dataloader=model.train_dataloader(),
        val_dataloaders=model.val_dataloader(),
    )
    assert result, "TPU doesn't work with dataloaders passed to fit()."


@pytest.mark.parametrize(['tpu_cores', 'expected_tpu_id'], [
    pytest.param(1, None),
    pytest.param(8, None),
    pytest.param([1], 1),
    pytest.param([8], 8),
])
def test_tpu_id_to_be_as_expected(tpu_cores, expected_tpu_id):
    """Test if trainer.tpu_id is set as expected"""
    assert Trainer(tpu_cores=tpu_cores).tpu_id == expected_tpu_id


@patch('pytorch_lightning.trainer.trainer.XLA_AVAILABLE', False)
def test_exception_when_no_tpu_found(tmpdir):
    """Test if exception is thrown when xla devices are not available"""
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        tpu_cores=8,
    )

    with pytest.raises(MisconfigurationException, match='No TPU devices found.'):
        trainer.fit(model)
