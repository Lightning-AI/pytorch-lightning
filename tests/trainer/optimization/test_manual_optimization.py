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
import torch
import pytest
import collections
from unittest import mock
from tests.base.boring_model import BoringModel, RandomDataset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import APEX_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException


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


class ExtendedModel(BoringModel):

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

        loss = 0.1 * self.loss(batch, output)
        if self.should_update:
            weight_before = self.layer.weight.clone()
            self.manual_backward(loss.clone(), opt)
            loss.detach()
            self.trainer.scaler.unscale_(opt)

            assert torch.sum(self.layer.weight.grad) != 0
            opt.step()

            after_before = self.layer.weight.clone()
            mask = torch.logical_and(torch.isnan(after_before), torch.isinf(after_before))
            assert not torch.equal(weight_before, after_before)
            opt.zero_grad()

        return loss.detach() if self.detach else loss

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        self.called["on_train_batch_end"] += 1
        after_before = self.layer.weight.clone()
        if self.should_update:
            assert not torch.equal(self.weight_before, after_before)
        else:
            assert torch.equal(self.weight_before, after_before)
        assert torch.sum(self.layer.weight.grad) == 0
        self.count += 1

    def on_train_end(self):
        assert self.called["training_step"] == 10
        assert self.called["on_train_batch_start"] == 10
        assert self.called["on_train_batch_end"] == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_automatic_optimization_false_and_return_tensor(tmpdir):
    """
    This test verify that in `automatic_optimization` we don't add gradient if the user return loss.
    """

    model = ExtendedModel()
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
def test_automatic_optimization_false_atest_automatic_optimization_falsend_return_detached_tensor(tmpdir):
    """
    This test verify that in `automatic_optimization`
    we don't add gradient if the user return loss + raise an error
    """

    model = ExtendedModel()
    model.detach = True
    model.training_step_end = None
    model.training_epoch_end = None

    match = "In manual optimization, `training_step` should not return a Tensor"
    with pytest.raises(MisconfigurationException, match=match):
        trainer = Trainer(
            max_epochs=1,
            default_root_dir=tmpdir,
            limit_train_batches=10,
            limit_test_batches=0,
            limit_val_batches=0,
            automatic_optimization=False,
            precision=16,
            amp_backend='native',
            accelerator="ddp",
            gpus=2,
        )
        trainer.fit(model)
