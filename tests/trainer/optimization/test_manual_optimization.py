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

import pytest
import torch

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.utilities import APEX_AVAILABLE
from tests.base.boring_model import BoringModel


def test_multiple_optimizers_manual(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

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


def test_multiple_optimizers_manual_return(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

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


def test_multiple_optimizers_manual_return_and_log(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_multiple_optimizers_manual_native_amp(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(not APEX_AVAILABLE, reason="test requires apex")
def test_multiple_optimizers_manual_apex(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_multiple_optimizers_manual_optimizer_step(tmpdir):
    os.environ['PL_DEV_DEBUG'] = '1'

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
