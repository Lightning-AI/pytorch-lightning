import os
import torch
import pytest
from tests.base.boring_model import BoringModel, RandomDataset
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import APEX_AVAILABLE


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

            self.backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self.step(batch[0])

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.backward(loss_2, opt_b, retain_graph=True)
            self.backward(loss_2, opt_a, retain_graph=True)

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
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)

    num_manual_backward_calls = 3
    assert len(trainer.dev_debugger.backward_calls) == limit_train_batches * num_manual_backward_calls


def test_multiple_optimizers_manual_single_optimizer_called(tmpdir):
    """
    Tests that during manual opt, the user may only call an optimizer once
    """
    os.environ['PL_DEV_DEBUG'] = '1'

    class TestModel(BoringModel):
        def training_step(self, batch, batch_idx, optimizer_idx):
            # manual
            (opt_a, opt_b) = self.optimizers()
            loss_1 = self.step(batch[0])

            # make sure there are no grads
            if batch_idx > 0:
                assert torch.all(self.layer.weight.grad == 0)

            self.backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self.step(batch[0])

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.backward(loss_2, opt_a, retain_graph=True)
            self.backward(loss_2, opt_a, retain_graph=True)

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
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
    )

    trainer.fit(model)

    assert len(trainer.dev_debugger.backward_calls) == limit_train_batches * 3 * 2


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

            self.backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self(batch[0])
            loss_2 = self.loss(loss_2, loss_2)

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.backward(loss_2, opt_b, retain_graph=True)
            self.backward(loss_2, opt_a, retain_graph=True)

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
        default_root_dir=tmpdir,
        limit_train_batches=limit_train_batches,
        limit_val_batches=2,
        max_epochs=1,
        log_every_n_steps=1,
        weights_summary=None,
        precision=16,
        amp_backend='apex',
        gpus=1
    )

    trainer.fit(model)

    num_manual_backward_calls = 3
    assert len(trainer.dev_debugger.backward_calls) == limit_train_batches * num_manual_backward_calls

    for call in trainer.dev_debugger.backward_calls:
        assert call['type'] == 'apex'


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

            self.backward(loss_1, opt_a)
            opt_a.step()
            opt_a.zero_grad()
            assert torch.all(self.layer.weight.grad == 0)

            # fake discriminator
            loss_2 = self.step(batch[0])

            # ensure we forward the correct params to the optimizer
            # without retain_graph we can't do multiple backward passes
            self.backward(loss_2, opt_b, retain_graph=True)
            self.backward(loss_2, opt_a, retain_graph=True)

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
    assert len(trainer.dev_debugger.backward_calls) == limit_train_batches * num_manual_backward_calls

    for call in trainer.dev_debugger.backward_calls:
        assert call['type'] == 'native_amp'
