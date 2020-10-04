import os
from pytorch_lightning import Trainer, seed_everything
from tests.base import EvalModelTemplate


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
