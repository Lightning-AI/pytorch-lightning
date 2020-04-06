import pytest

import torch
import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import (
    LightTrainDataloader,
    TestModelBase,
    LightTestMultipleOptimizersWithSchedulingMixin,
)


def test_error_on_more_than_1_optimizer(tmpdir):
    ''' Check that error is thrown when more than 1 optimizer is passed '''
    tutils.reset_seed()

    class CurrentTestModel(
        LightTestMultipleOptimizersWithSchedulingMixin,
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1
    )

    with pytest.raises(MisconfigurationException):
        trainer.find_lr(model)


def test_model_reset_correctly(tmpdir):
    ''' Check that model weights are correctly reset after find_lr() '''
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1
    )

    before_state_dict = model.state_dict()

    _ = trainer.find_lr(model, num_training=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), \
            'Model was not reset correctly after learning rate finder'


def test_trainer_reset_correctly(tmpdir):
    ''' Check that all trainer parameters are reset correctly after find_lr() '''
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1
    )

    changed_attributes = ['callbacks', 'logger', 'max_steps', 'auto_lr_find',
                          'progress_bar_refresh_rate',
                          'accumulate_grad_batches',
                          'checkpoint_callback']
    attributes_before = {}
    for ca in changed_attributes:
        attributes_before[ca] = getattr(trainer, ca)

    _ = trainer.find_lr(model, num_training=5)

    attributes_after = {}
    for ca in changed_attributes:
        attributes_after[ca] = getattr(trainer, ca)

    for key in changed_attributes:
        assert attributes_before[key] == attributes_after[key], \
            f'Attribute {key} was not reset correctly after learning rate finder'


def test_full_basic_run(tmpdir):
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)
    before_lr = hparams.learning_rate
    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1,
        auto_lr_find=True
    )

    trainer.fit(model)
    after_lr = model.hparams.learning_rate
    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'


def test_full_power_run(tmpdir):
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)
    before_lr = hparams.learning_rate
    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1,
    )

    lrfinder = trainer.find_lr(model)
    after_lr = lrfinder.suggestion()
    model.hparams.learning_rate = after_lr
    trainer.fit(model)

    assert before_lr != after_lr, \
        'Learning rate was not altered after running learning rate finder'
