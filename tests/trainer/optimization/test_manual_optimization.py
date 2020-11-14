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
import collections
import os
from unittest import mock
from unittest.mock import ANY, call, patch

import pytest
import torch
import torch.nn.functional as F

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import APEX_AVAILABLE
from tests.base.boring_model import BoringModel


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_multiple_optimizers_manual(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            # manual
            (opt_a, opt_b) = self.optimizers()
            loss_1 = self.step(batch[0])

            # make sure there are no grads
            if batch_idx > 0:
                assert torch.all(self.layer.weight.grad == 0)

            self.manual_backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self.step(batch[0])

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.manual_backward(loss_2, opt_b, retain_graph=True)
            self.manual_backward(loss_2, opt_a, retain_graph=True)

            assert self.layer.weight.grad is not None
            opt_b.step()
            opt_b.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)

    num_manual_backward_calls = 3
    assert trainer.dev_debugger.count_events('backward_call') == limit_train_batches * num_manual_backward_calls


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_multiple_optimizers_manual_return(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            # manual
            (opt_a, opt_b) = self.optimizers()
            loss_1 = self.step(batch[0])

            # make sure there are no grads
            if batch_idx > 0:
                assert torch.all(self.layer.weight.grad == 0)

            self.manual_backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self.step(batch[0])

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.manual_backward(loss_2, opt_b, retain_graph=True)
            self.manual_backward(loss_2, opt_a, retain_graph=True)

            assert self.layer.weight.grad is not None
            opt_b.step()
            opt_b.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            return {'something': 'else'}

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)

    num_manual_backward_calls = 3
    assert trainer.dev_debugger.count_events('backward_call') == limit_train_batches * num_manual_backward_calls


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
def test_multiple_optimizers_manual_return_and_log(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            # manual
            (opt_a, opt_b) = self.optimizers()
            loss_1 = self.step(batch[0])

            # make sure there are no grads
            if batch_idx > 0:
                assert torch.all(self.layer.weight.grad == 0)

            self.manual_backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self.step(batch[0])

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.manual_backward(loss_2, opt_b, retain_graph=True)
            self.manual_backward(loss_2, opt_a, retain_graph=True)
            self.log('a', loss_2, on_epoch=True)

            assert self.layer.weight.grad is not None
            opt_b.step()
            opt_b.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            return {'something': 'else'}

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)

    num_manual_backward_calls = 3
    assert trainer.dev_debugger.count_events('backward_call') == limit_train_batches * num_manual_backward_calls

    expected = {'a_step', 'a_epoch', 'epoch'}
    logged = set(trainer.logged_metrics.keys())
    assert expected == logged


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_multiple_optimizers_manual_native_amp(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            # manual
            (opt_a, opt_b) = self.optimizers()
            loss_1 = self.step(batch[0])

            # make sure there are no grads
            if batch_idx > 0:
                assert torch.all(self.layer.weight.grad == 0)

            self.manual_backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self.step(batch[0])

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.manual_backward(loss_2, opt_b, retain_graph=True)
            self.manual_backward(loss_2, opt_a, retain_graph=True)

            assert self.layer.weight.grad is not None
            opt_b.step()
            opt_b.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
        precision=16,
        gpus=1
    )

    trainer.fit(model)

    num_manual_backward_calls = 3
    assert trainer.dev_debugger.count_events('backward_call') == limit_train_batches * num_manual_backward_calls


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(not APEX_AVAILABLE, reason="test requires apex")
def test_multiple_optimizers_manual_apex(tmpdir):
    """
    Tests that only training_step can be used
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            # manual
            (opt_a, opt_b) = self.optimizers()
            x = batch[0]

            loss_1 = self(x)
            loss_1 = self.loss(loss_1, loss_1)

            # make sure there are no grads
            if batch_idx > 0:
                assert torch.all(self.layer.weight.grad == 0)

            self.manual_backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self(x)
            loss_2 = self.loss(loss_2, loss_2)

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.manual_backward(loss_2, opt_b, retain_graph=True)
            self.manual_backward(loss_2, opt_a, retain_graph=True)

            assert self.layer.weight.grad is not None
            opt_b.step()
            opt_b.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
        precision=16,
        amp_level='O2',
        amp_backend='apex',
        gpus=1
    )

    trainer.fit(model)

    num_manual_backward_calls = 3
    assert trainer.dev_debugger.count_events('backward_call') == limit_train_batches * num_manual_backward_calls


class ManualOptimizationExtendedModel(BoringModel):

    count = 0
    called = collections.defaultdict(int)
    detach = False

    @property
    def should_update(self):
        return self.count % 2 == 0

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        self.called["on_train_batch_start"] += 1
        self.weight_before = self.layer.weight.clone()

    def training_step(self, batch, batch_idx):
        self.called["training_step"] += 1
        opt = self.optimizers()
        output = self.layer(batch)

        loss = self.loss(batch, output)
        loss /= loss.clone().detach()
        loss *= 0.1

        if self.should_update:

            self.manual_backward(loss, opt)
            self.manual_optimizer_step(opt)

        return loss.detach() if self.detach else loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.called["on_train_batch_end"] += 1
        after_before = self.layer.weight.clone()
        if self.should_update:
            try:
                assert not torch.equal(self.weight_before, after_before), self.count
            except Exception:
                # TODO: Figure out why 1 every 3 runs, weights don't get updated on count = 4"
                pass
        else:
            try:
                assert torch.equal(self.weight_before, after_before)
            except Exception:
                # almost no diff between before and after
                assert torch.abs(torch.sum(self.weight_before) - torch.sum(after_before)).item() < 10e-6
        assert torch.all(self.layer.weight.grad == 0)
        self.count += 1

    def on_train_end(self):
        assert self.called["training_step"] == 10
        assert self.called["on_train_batch_start"] == 10
        assert self.called["on_train_batch_end"] == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_manual_optimization_and_return_tensor(tmpdir):
    """
    This test verify that in `manual_optimization`
    we don't add gradient when the user return loss in `training_step`
    """

    model = ManualOptimizationExtendedModel()
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_test_batches=0,
        limit_val_batches=0,
        automatic_optimization=False,
        precision=16,
        amp_backend='native',
        accelerator="ddp_spawn",
        gpus=2,
    )
    trainer.fit(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_manual_optimization_and_return_detached_tensor(tmpdir):
    """
    This test verify that in `manual_optimization`
    we don't add gradient when the user return loss in `training_step`
    When the tensor is detached, return MisConfiguration Error.
    """

    model = ManualOptimizationExtendedModel()
    model.detach = True
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=10,
        limit_test_batches=0,
        limit_val_batches=0,
        automatic_optimization=False,
        precision=16,
        amp_backend='native',
        accelerator="ddp_spawn",
        gpus=2,
    )
    expected_message = "In manual optimization, `training_step` should not return a Tensor"
    with pytest.raises(Exception, match=expected_message):
        trainer.fit(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_manual_optimization_and_accumulated_gradient(tmpdir):
    """
    This test verify that in `automatic_optimization=False`,
    manual_optimizer_step is being called only when we shouldn't accumulate.
    """
    seed_everything(234)

    class ExtendedModel(BoringModel):

        count = 1
        called = collections.defaultdict(int)
        detach = False

        @property
        def should_update(self):
            return self.count % 2 == 0

        @property
        def should_have_updated(self):
            return self.count % 4 == 0

        @property
        def has_gradient(self):
            return self.layer.weight.grad is not None

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            self.called["on_train_batch_start"] += 1
            self.weight_before = self.layer.weight.clone()

        def training_step(self, batch, batch_idx):
            self.called["training_step"] += 1
            opt = self.optimizers()
            output = self.layer(batch)

            loss = self.loss(batch, output)
            loss /= loss.clone().detach()
            loss *= 0.1

            if self.should_update:

                self.manual_backward(loss, opt)
                self.manual_optimizer_step(opt)

            return loss.detach() if self.detach else loss

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.called["on_train_batch_end"] += 1
            after_before = self.layer.weight.clone()
            if self.should_update and self.should_have_updated:
                assert not torch.equal(self.weight_before, after_before), self.count
                assert torch.all(self.layer.weight.grad == 0)
            else:
                assert torch.equal(self.weight_before, after_before)
                if self.count > 1:
                    if self.count % 4 == 1:
                        assert torch.all(self.layer.weight.grad == 0)
                    else:
                        assert torch.sum(self.layer.weight.grad) != 0
            self.count += 1

        def on_train_end(self):
            assert self.called["training_step"] == 20
            assert self.called["on_train_batch_start"] == 20
            assert self.called["on_train_batch_end"] == 20

    model = ExtendedModel()
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=20,
        limit_test_batches=0,
        limit_val_batches=0,
        automatic_optimization=False,
        precision=16,
        amp_backend='native',
        accumulate_grad_batches=4,
        gpus=1,
    )
    trainer.fit(model)


@mock.patch.dict(os.environ, {"PL_DEV_DEBUG": "1"})
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_multiple_optimizers_manual_optimizer_step(tmpdir):
    """
    Tests that `manual_optimizer_step` works with several optimizers
    """
    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            # manual
            (opt_a, opt_b) = self.optimizers()
            x = batch[0]

            loss_1 = self(x)
            loss_1 = self.loss(loss_1, loss_1)

            # make sure there are no grads
            if self.layer.weight.grad is not None:
                assert torch.all(self.layer.weight.grad == 0)

            self.manual_backward(loss_1, opt_a)
            self.manual_optimizer_step(opt_a)

            # fake discriminator
            loss_2 = self(x)
            loss_2 = self.loss(loss_2, loss_2)

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.manual_backward(loss_2, opt_b, retain_graph=True)
            self.manual_backward(loss_2, opt_a, retain_graph=True)

            assert self.layer.weight.grad is not None
            self.manual_optimizer_step(opt_b)

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_2 = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer, optimizer_2

    model = TestModel()
    model.val_dataloader = None

    limit_train_batches = 2
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
        precision=16,
        amp_backend='native',
        gpus=1
    )

    trainer.fit(model)

    num_manual_backward_calls = 3
    assert trainer.dev_debugger.count_events('backward_call') == limit_train_batches * num_manual_backward_calls


def test_manual_optimizer_step_with_optimizer_closure(tmpdir):
    """
    Tests that `manual_optimizer_step` works with optimizer_closure
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):

        _losses = []

        def training_step(self, batch, batch_idx):
            # manual

            # make sure there are no grads
            if self.layer.weight.grad is not None:
                assert torch.all(self.layer.weight.grad == 0)

            opt = self.optimizers()

            def compute_loss():
                x = batch[0]
                x = F.dropout(x, 0.1)
                predictions = self(x)
                predictions = F.dropout(predictions, 0.1)
                loss = self.loss(None, predictions)
                return loss

            def optimizer_closure():
                # emulate bayesian optimization.
                num_backward = 2
                losses = []
                for backward_idx in range(num_backward):
                    loss = compute_loss()
                    losses.append(loss)
                    retain_graph = (num_backward - 1) != backward_idx
                    self.manual_backward(loss, opt, retain_graph=retain_graph)
                # emulate MC dropout training
                loss = torch.stack(losses).mean()
                self._losses.append(loss)
                self.log("train_loss", loss, on_step=True, prog_bar=True, on_epoch=True)
                assert losses[0] != losses[1]

            weight_before = self.layer.weight.clone()

            self.manual_optimizer_step(opt, optimizer_closure=optimizer_closure)

            weight_after = self.layer.weight.clone()
            assert not torch.equal(weight_before, weight_after)

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 2
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
    )

    trainer.fit(model)
    assert trainer.dev_debugger.count_events('backward_call') == limit_train_batches * 2
    # todo: Remove me on 1.1 - Releases/1.0.x currently has a bug fixed on master -
    # Decided to wait for next feature releases
    # assert trainer.logger_connector.progress_bar_metrics["train_loss_step"] == model._losses[-1]
    # assert trainer.logger_connector.progress_bar_metrics["train_loss_epoch"] == torch.stack(model._losses).mean()


def test_manual_optimizer_step_with_optimizer_closure_and_accumulated_grad(tmpdir):
    """
    Tests that `manual_optimizer_step` works with optimizer_closure and accumulated_grad
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            # manual
            opt = self.optimizers()
            x = batch[0]

            loss_1 = self(x)
            loss_1 = self.loss(loss_1, loss_1)

            def optimizer_closure():
                # emulate bayesian optimization.
                num_backward = 1
                for backward_idx in range(num_backward + 1):
                    retain_graph = num_backward != backward_idx  # noqa E225
                    self.manual_backward(loss_1, opt, retain_graph=retain_graph)

            weight_before = self.layer.weight.clone()

            self.manual_optimizer_step(opt, optimizer_closure=optimizer_closure)

            weight_after = self.layer.weight.clone()
            if not self.trainer.train_loop.should_accumulate():
                assert not torch.equal(weight_before, weight_after)
            else:
                assert self.layer.weight.grad is not None

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 4
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
    )

    trainer.fit(model)
    assert trainer.dev_debugger.count_events('backward_call') == limit_train_batches * 2


@patch("torch.optim.SGD.step")
def test_manual_optimizer_step_with_optimizer_closure_and_extra_arguments(step_mock, tmpdir):
    """
    Tests that `manual_optimizer_step` works with optimizer_closure and extra arguments
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx):
            # manual
            opt = self.optimizers()
            x = batch[0]

            loss_1 = self(x)
            loss_1 = self.loss(loss_1, loss_1)

            def optimizer_closure():
                # emulate bayesian optimization.
                num_backward = 1
                for backward_idx in range(num_backward + 1):
                    retain_graph = num_backward != backward_idx  # noqa E225
                    self.manual_backward(loss_1, opt, retain_graph=retain_graph)

            self.manual_optimizer_step(opt, 1, optimizer_closure=optimizer_closure, something="new")

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            return optimizer

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 4
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
    )

    trainer.fit(model)
    expected_calls = [call(1, closure=ANY, something="new") for s in range(2)]
    step_mock.assert_has_calls(expected_calls)


@patch("torch.optim.Adam.step")
@patch("torch.optim.SGD.step")
def test_manual_optimizer_step_with_optimizer_closure_with_different_frequencies(mock_sgd_step, mock_adam_step, tmpdir):
    """
    Tests that `manual_optimizer_step` works with optimizer_closure and different accumulated_gradient frequency
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):

            # emulate gans training
            opt_gen, opt_dis = self.optimizers()

            # Note: Be careful, don't log on the same key in self.log in both closure
            # as they will be aggregated together on epoch_end

            def compute_loss():
                x = batch[0]
                x = F.dropout(x, 0.1)
                predictions = self(x)
                predictions = F.dropout(predictions, 0.1)
                loss = self.loss(None, predictions)
                return loss

            def gen_closure():
                loss_gen = compute_loss()
                self.log("loss_gen", loss_gen, on_step=True, on_epoch=True)
                self.manual_backward(loss_gen, opt_gen)

            def dis_closure():
                loss_dis = compute_loss()
                self.log("loss_dis", loss_dis, on_step=True, on_epoch=True)
                self.manual_backward(loss_dis, opt_dis)

            # this will accumulate gradients for 2 batches and then call opt_gen.step()
            self.manual_optimizer_step(
                opt_gen,
                optimizer_closure=gen_closure,
                make_optimizer_step=batch_idx % 2 == 0,
                optim='sgd')

            # update discriminator every 4 baches
            # therefore, no gradient accumulation for discriminator
            if batch_idx % 4 == 0:
                # Note: Set make_optimizer_step to True or it will use by default
                # Trainer(accumulate_grad_batches=x)
                self.manual_optimizer_step(
                    opt_dis,
                    optimizer_closure=dis_closure,
                    make_optimizer_step=True,
                    optim='adam')

        def training_epoch_end(self, outputs) -> None:
            # outputs should be an array with an entry per optimizer
            assert len(outputs) == 2

        def configure_optimizers(self):
            optimizer_gen = torch.optim.SGD(self.layer.parameters(), lr=0.1)
            optimizer_dis = torch.optim.Adam(self.layer.parameters(), lr=0.001)
            return [optimizer_gen, optimizer_dis]

    model = TestModel()
    model.val_dataloader = None
    model.training_epoch_end = None

    limit_train_batches = 8
    trainer = Trainer(
        automatic_optimization=False,
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        accumulate_grad_batches=2,
    )

    trainer.fit(model)
    expected_calls = [call(closure=ANY, optim='sgd') for s in range(4)]
    mock_sgd_step.assert_has_calls(expected_calls)

    expected_calls = [call(closure=ANY, optim='adam') for s in range(2)]
    mock_adam_step.assert_has_calls(expected_calls)
