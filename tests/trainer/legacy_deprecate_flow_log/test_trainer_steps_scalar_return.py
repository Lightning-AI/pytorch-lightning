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
import os
from unittest import mock

import pytest
import torch

from pytorch_lightning import Trainer
from tests.helpers import BoringModel
from tests.helpers.deterministic_model import DeterministicModel


def test_training_step_scalar(tmpdir):
    """
    Tests that only training_step that returns a single scalar can be used
    """
    model = DeterministicModel()
    model.training_step = model.training_step__scalar_return
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
    assert len(out.grad_norm_dic) == 0 and isinstance(out.grad_norm_dic, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out['minimize'], torch.Tensor)
    assert train_step_out['minimize'].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens
    )
    assert opt_closure_result['loss'].item() == 171


def training_step_scalar_with_step_end(tmpdir):
    """
    Checks train_step with scalar only + training_step_end
    """
    model = DeterministicModel()
    model.training_step = model.training_step__scalar_return
    model.training_step_end = model.training_step_end__scalar
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
    assert len(out.grad_norm_dic) == 0 and isinstance(out.grad_norm_dic, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out, torch.Tensor)
    assert train_step_out.item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens
    )
    assert opt_closure_result['loss'].item() == 171


def test_full_training_loop_scalar(tmpdir):
    """
    Checks train_step + training_step_end + training_epoch_end
    (all with scalar return from train_step)
    """

    model = DeterministicModel()
    model.training_step = model.training_step__scalar_return
    model.training_step_end = model.training_step_end__scalar
    model.training_epoch_end = model.training_epoch_end__scalar
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
    assert len(out.grad_norm_dic) == 0 and isinstance(out.grad_norm_dic, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out['minimize'], torch.Tensor)
    assert train_step_out['minimize'].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens
    )
    assert opt_closure_result['loss'].item() == 171


def test_train_step_epoch_end_scalar(tmpdir):
    """
    Checks train_step + training_epoch_end (NO training_step_end)
    (with scalar return)
    """

    model = DeterministicModel()
    model.training_step = model.training_step__scalar_return
    model.training_step_end = None
    model.training_epoch_end = model.training_epoch_end__scalar
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
    assert len(out.grad_norm_dic) == 0 and isinstance(out.grad_norm_dic, dict)

    train_step_out = out.training_step_output_for_epoch_end
    assert len(train_step_out) == 1
    train_step_out = train_step_out[0][0]
    assert isinstance(train_step_out['minimize'], torch.Tensor)
    assert train_step_out['minimize'].item() == 171

    # make sure the optimizer closure returns the correct things
    opt_closure_result = trainer.train_loop.training_step_and_backward(
        batch, batch_idx, 0, trainer.optimizers[0], trainer.hiddens
    )
    assert opt_closure_result['loss'].item() == 171


class DPPReduceMeanPbarModel(BoringModel):

    logged = []

    def training_step(self, batch, batch_idx):
        output = self.layer(batch)
        loss = self.loss(batch, output)
        loss /= loss.clone().detach()
        self.log('self_log', loss, prog_bar=True, sync_dist=True)
        return {"loss": loss, "progress_bar": {"loss_2": loss}}


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_dpp_reduce_mean_pbar(tmpdir):

    model = DPPReduceMeanPbarModel()
    model.training_step_end = None
    model.training_epoch_end = None

    distributed_backend = "ddp_spawn"

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=os.getcwd(),
        limit_train_batches=10,
        limit_test_batches=2,
        limit_val_batches=2,
        accelerator=distributed_backend,
        gpus=2,
        precision=32
    )

    trainer.fit(model)

    # TODO: Move this test to DDP. pbar_added_metrics is empty with ddp_spawn for some reasons

    pbar_added_metrics = trainer.dev_debugger.pbar_added_metrics
    is_in = False
    for pbar_metrics in pbar_added_metrics:
        if 'loss_2' in pbar_metrics:
            is_in = True
            assert pbar_metrics["loss_2"].item() == 1
    if distributed_backend == "ddp":
        assert is_in is True
