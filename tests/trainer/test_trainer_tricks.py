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


def test_model_reset_correctly(tmpdir):
    """ Check that model weights are correctly reset after scaling batch size. """
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

    trainer.scale_batch_size(model, n_max_try=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), \
            'Model was not reset correctly after scaling batch size'


def test_trainer_reset_correctly(tmpdir):
    ''' Check that all trainer parameters are reset correctly after scaling batch size'''
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

    changed_attributes = ['max_steps', 'weights_summary', 'logger',
                          'callbacks', 'checkpoint_callback']

    attributes_before = {}
    for ca in changed_attributes:
        attributes_before[ca] = getattr(trainer, ca)

    trainer.scale_batch_size(model, n_max_try=5)

    attributes_after = {}
    for ca in changed_attributes:
        attributes_after[ca] = getattr(trainer, ca)

    for key in changed_attributes:
        assert attributes_before[key] == attributes_after[key], \
            f'Attribute {key} was not reset correctly after learning rate finder'


def test_trainer_arg_bool(tmpdir):
    ''' Check that trainer arg works with bool input '''
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)
    before_batch_size = hparams.batch_size
    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1,
        auto_scale_batch_size=True
    )

    trainer.fit(model)
    after_batch_size = model.hparams.batch_size
    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'


def test_trainer_arg_str(tmpdir):
    ''' Check that trainer arg works with str input '''
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)
    before_batch_size = hparams.batch_size
    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1,
        auto_scale_batch_size='binsearch'
    )

    trainer.fit(model)
    after_batch_size = model.hparams.batch_size
    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'


def test_call_to_trainer_method(tmpdir):
    ''' Test that calling the trainer method itself works '''
    tutils.reset_seed()

    class CurrentTestModel(
        LightTrainDataloader,
        TestModelBase,
    ):
        pass

    hparams = tutils.get_default_hparams()
    model = CurrentTestModel(hparams)
    before_batch_size = hparams.batch_size
    # logger file to get meta
    trainer = Trainer(
        default_save_path=tmpdir,
        max_epochs=1,
    )

    after_batch_size = trainer.scale_batch_size(model, mode='binsearch')
    model.hparams.batch_size = after_batch_size
    trainer.fit(model)

    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'
