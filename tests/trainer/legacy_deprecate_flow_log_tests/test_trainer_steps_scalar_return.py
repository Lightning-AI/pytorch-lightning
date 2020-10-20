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
"""
Tests to ensure that the training loop works with a scalar
"""
import torch

from pytorch_lightning import Trainer
from tests.base.deterministic_model import DeterministicModel


def test_training_step_scalar(tmpdir):
    """
    Tests that only training_step that returns a single scalar can be used
    """
    model = DeterministicModel()
    model.training_step = model.training_step_scalar_return
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert not model.training_epoch_end_called

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.train_loop.run_training_batch(batch, batch_idx, 0)
    assert out.signal == 0
    assert len(out.batch_log_metrics) == 0 and isinstance(out.batch_log_metrics, dict)
    assert len(out.grad_norm_dic) == 0 and isinstance(out.grad_norm_dic, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out['minimize'], torch.Tensor)
    assert train_step_out['minimize'].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'].item() == 171


def training_step_scalar_with_step_end(tmpdir):
    """
    Checks train_step with scalar only + training_step_end
    """
    model = DeterministicModel()
    model.training_step = model.training_step_scalar_return
    model.training_step_end = model.training_step_end_scalar
    model.val_dataloader = None

    trainer = Trainer(fast_dev_run=True, weights_summary=None)
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert model.training_step_end_called
    assert not model.training_epoch_end_called

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.train_loop.run_training_batch(batch, batch_idx, 0)
    assert out.signal == 0
    assert len(out.batch_log_metrics) == 0 and isinstance(out.batch_log_metrics, dict)
    assert len(out.grad_norm_dic) == 0 and isinstance(out.grad_norm_dic, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out, torch.Tensor)
    assert train_step_out.item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'].item() == 171


def test_full_training_loop_scalar(tmpdir):
    """
    Checks train_step + training_step_end + training_epoch_end
    (all with scalar return from train_step)
    """
    model = DeterministicModel()
    model.training_step = model.training_step_scalar_return
    model.training_step_end = model.training_step_end_scalar
    model.training_epoch_end = model.training_epoch_end_scalar
    model.val_dataloader = None

    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        weights_summary=None,
    )
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert model.training_step_end_called
    assert model.training_epoch_end_called

    # assert epoch end metrics were added
    assert len(trainer.logger_connector.callback_metrics) == 0
    assert len(trainer.logger_connector.progress_bar_metrics) == 0

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.train_loop.run_training_batch(batch, batch_idx, 0)
    assert out.signal == 0
    assert len(out.batch_log_metrics) == 0 and isinstance(out.batch_log_metrics, dict)
    assert len(out.grad_norm_dic) == 0 and isinstance(out.grad_norm_dic, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out['minimize'], torch.Tensor)
    assert train_step_out['minimize'].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'].item() == 171


def test_train_step_epoch_end_scalar(tmpdir):
    """
    Checks train_step + training_epoch_end (NO training_step_end)
    (with scalar return)
    """
    model = DeterministicModel()
    model.training_step = model.training_step_scalar_return
    model.training_step_end = None
    model.training_epoch_end = model.training_epoch_end_scalar
    model.val_dataloader = None

    trainer = Trainer(max_epochs=1, weights_summary=None)
    trainer.fit(model)

    # make sure correct steps were called
    assert model.training_step_called
    assert not model.training_step_end_called
    assert model.training_epoch_end_called

    # assert epoch end metrics were added
    assert len(trainer.logger_connector.callback_metrics) == 0
    assert len(trainer.logger_connector.progress_bar_metrics) == 0

    # make sure training outputs what is expected
    for batch_idx, batch in enumerate(model.train_dataloader()):
        break

    out = trainer.train_loop.run_training_batch(batch, batch_idx, 0)
    assert out.signal == 0
    assert len(out.batch_log_metrics) == 0 and isinstance(out.batch_log_metrics, dict)
    assert len(out.grad_norm_dic) == 0 and isinstance(out.grad_norm_dic, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out['minimize'], torch.Tensor)
    assert train_step_out['minimize'].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens)
    assert opt_closure_result['loss'].item() == 171
