import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from torch.utils.data import RandomSampler


def test_overfit(tmpdir):

    model = EvalModelTemplate()
    model.train_dataloader()

    # original train loader which should be replaced in all methods
    train_loader = model.train_dataloader()
    full_train_samples = len(train_loader)
    num_train_samples = int(0.11 * full_train_samples)

    # make sure the original loader has shuffle=True
    assert isinstance(train_loader.sampler, RandomSampler)

    # pull out a batch from the original loader
    (xa, ya) = next(iter(train_loader))

    # make sure the train set is also the val and test set
    for split in ['val', 'test']:
        # test percent overfit batches
        loader_num_batches, dataloaders = Trainer(overfit_batches=0.11)._reset_eval_dataloader(model, split)
        assert loader_num_batches == num_train_samples
        assert isinstance(dataloaders[0].sampler, RandomSampler)

        # assert that it's the same dataloader
        (xb, yb) = next(iter(dataloaders[0]))
        assert torch.eq(xa, xb)
        assert torch.eq(ya, yb)

        # test overfit number of batches
        loader_num_batches, dataloaders = Trainer(overfit_batches=1)._reset_eval_dataloader(model, split)
        assert loader_num_batches == 1
        loader_num_batches, dataloaders = Trainer(overfit_batches=5)._reset_eval_dataloader(model, split)
        assert loader_num_batches == 5

        # make sure the samplers are disabled
        assert isinstance(dataloaders[0].sampler, RandomSampler)

        # assert that it's the same dataloader
        (xb, yb) = next(iter(dataloaders[0]))
        assert torch.eq(xa, xb)
        assert torch.eq(ya, yb)




def test_model_reset_correctly(tmpdir):
    """ Check that model weights are correctly reset after scaling batch size. """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )

    before_state_dict = model.state_dict()

    trainer.scale_batch_size(model, max_trials=5)

    after_state_dict = model.state_dict()

    for key in before_state_dict.keys():
        assert torch.all(torch.eq(before_state_dict[key], after_state_dict[key])), \
            'Model was not reset correctly after scaling batch size'


def test_trainer_reset_correctly(tmpdir):
    """ Check that all trainer parameters are reset correctly after scaling batch size. """
    tutils.reset_seed()

    model = EvalModelTemplate()

    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1
    )

    changed_attributes = ['max_steps',
                          'weights_summary',
                          'logger',
                          'callbacks',
                          'checkpoint_callback',
                          'early_stop_callback',
                          'enable_early_stop',
                          'train_percent_check']

    attributes_before = {}
    for ca in changed_attributes:
        attributes_before[ca] = getattr(trainer, ca)

    trainer.scale_batch_size(model, max_trials=5)

    attributes_after = {}
    for ca in changed_attributes:
        attributes_after[ca] = getattr(trainer, ca)

    for key in changed_attributes:
        assert attributes_before[key] == attributes_after[key], \
            f'Attribute {key} was not reset correctly after learning rate finder'


@pytest.mark.parametrize('scale_arg', ['power', 'binsearch'])
def test_trainer_arg(tmpdir, scale_arg):
    """ Check that trainer arg works with bool input. """
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    before_batch_size = hparams.get('batch_size')
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        auto_scale_batch_size=scale_arg,
    )

    trainer.fit(model)
    after_batch_size = model.batch_size
    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'


@pytest.mark.parametrize('scale_method', ['power', 'binsearch'])
def test_call_to_trainer_method(tmpdir, scale_method):
    """ Test that calling the trainer method itself works. """
    tutils.reset_seed()

    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    before_batch_size = hparams.get('batch_size')
    # logger file to get meta
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
    )

    after_batch_size = trainer.scale_batch_size(model, mode=scale_method, max_trials=5)
    model.batch_size = after_batch_size
    trainer.fit(model)

    assert before_batch_size != after_batch_size, \
        'Batch size was not altered after running auto scaling of batch size'


def test_error_on_dataloader_passed_to_fit(tmpdir):
    """Verify that when the auto scale batch size feature raises an error
       if a train dataloader is passed to fit """

    # only train passed to fit
    model = EvalModelTemplate()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_val_batches=0.1,
        train_percent_check=0.2,
        auto_scale_batch_size='power'
    )
    fit_options = dict(train_dataloader=model.dataloader(train=True))

    with pytest.raises(MisconfigurationException):
        trainer.fit(model, **fit_options)
