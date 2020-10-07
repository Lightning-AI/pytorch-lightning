import os
from pytorch_lightning import Trainer, seed_everything, callbacks
from tests.base import EvalModelTemplate, BoringModel
from unittest import mock
import pytest
import torch


os.environ['PL_DEV_DEBUG'] = '1'

def test_mc_called_on_fastdevrun(tmpdir):
    seed_everything(1234)
    os.environ['PL_DEV_DEBUG'] = '1'

    train_val_step_model = EvalModelTemplate()

    # fast dev run = called once
    # train loop only, dict, eval result
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(train_val_step_model)

    # checkpoint should have been called once with fast dev run
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 1

    # -----------------------
    # also called once with no val step
    # -----------------------
    train_step_only_model = EvalModelTemplate()
    train_step_only_model.validation_step = None

    # fast dev run = called once
    # train loop only, dict, eval result
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(train_step_only_model)

    # make sure only training step was called
    assert train_step_only_model.training_step_called
    assert not train_step_only_model.validation_step_called
    assert not train_step_only_model.test_step_called

    # checkpoint should have been called once with fast dev run
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 1


def test_mc_called(tmpdir):
    seed_everything(1234)
    os.environ['PL_DEV_DEBUG'] = '1'

    # -----------------
    # TRAIN LOOP ONLY
    # -----------------
    train_step_only_model = EvalModelTemplate()
    train_step_only_model.validation_step = None

    # no callback
    trainer = Trainer(max_epochs=3, checkpoint_callback=False)
    trainer.fit(train_step_only_model)
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 0

    # -----------------
    # TRAIN + VAL LOOP ONLY
    # -----------------
    val_train_model = EvalModelTemplate()
    # no callback
    trainer = Trainer(max_epochs=3, checkpoint_callback=False)
    trainer.fit(val_train_model)
    assert len(trainer.dev_debugger.checkpoint_callback_history) == 0


@mock.patch('torch.save')
@pytest.mark.parametrize(['epochs', 'val_check_interval', 'expected'],
                         [(1, 1.0, 1), (2, 1.0, 2), (1, 0.25, 4), (2, 0.3, 7)])
def test_default_checkpoint_freq(save_mock, tmpdir, epochs, val_check_interval, expected):

    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=epochs,
        weights_summary=None,
        val_check_interval=val_check_interval
    )
    trainer.fit(model)

    # make sure types are correct
    assert save_mock.call_count == expected


@mock.patch('torch.save')
@pytest.mark.parametrize(['k', 'epochs', 'val_check_interval', 'expected'],
                         [(1, 1, 1.0, 1), (2, 2, 1.0, 2), (2, 1, 0.25, 4), (2, 2, 0.3, 7)])
def test_top_k(save_mock, tmpdir, k, epochs, val_check_interval, expected):

    class TestModel(BoringModel):
        def __init__(self):
            super().__init__()
            self.last_coeff = 10.0

        def training_step(self, batch, batch_idx):
            loss = self.step(torch.ones(32))
            loss = loss / (loss + 0.0000001)
            loss += self.last_coeff
            self.log('my_loss', loss)
            self.last_coeff *= 0.999
            return loss

    model = TestModel()
    trainer = Trainer(
        checkpoint_callback=callbacks.ModelCheckpoint(monitor='my_loss', save_top_k=k),
        default_root_dir=tmpdir,
        max_epochs=epochs,
        weights_summary=None,
        val_check_interval=val_check_interval
    )
    trainer.fit(model)

    # make sure types are correct
    assert save_mock.call_count == expected
