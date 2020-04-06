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

"""
test_using_trainer_arg
test_using_lr_find
test_model_reset_correctly

"""


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
    ''' Check that all variables and methods are correctly reset after find_lr() '''
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
       
    lrfinder = trainer.find_lr(model, num_training=5)
    
    after_state_dict = model.state_dict()
    
    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key],after_state_dict[key])), \
            'Model was not reset correctly after learning rate finder'
