import os

import pytest
import torch

import tests.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)
from pytorch_lightning.testing import (
    LightningTestModel,
    LightningTestModelBase,
    LightningValidationStepMixin,
    LightningValidationMultipleDataloadersMixin,
    LightningTestMultipleDataloadersMixin,
)
from pytorch_lightning.trainer import training_io
from pytorch_lightning.trainer.logging import TrainerLoggingMixin


def test_no_val_module(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()

    class CurrentTestModel(LightningTestModelBase):
        pass

    model = CurrentTestModel(hparams)

    # logger file to get meta
    logger = tutils.get_test_tube_logger(tmpdir, False)

    trainer_options = dict(
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # training complete
    assert result == 1, 'amp + ddp model failed to complete'

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    tags_path = logger.experiment.get_data_path(logger.experiment.name, logger.experiment.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_metrics(weights_path=new_weights_path,
                                                   tags_csv=tags_path)
    model_2.eval()


def test_no_val_end_module(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    tutils.reset_seed()

    class CurrentTestModel(LightningValidationStepMixin, LightningTestModelBase):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    logger = tutils.get_test_tube_logger(tmpdir, False)

    trainer_options = dict(
        max_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    tags_path = logger.experiment.get_data_path(logger.experiment.name, logger.experiment.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_metrics(weights_path=new_weights_path,
                                                   tags_csv=tags_path)
    model_2.eval()


def test_gradient_accumulation_scheduling(tmpdir):
    tutils.reset_seed()

    """
    Test grad accumulation by the freq of optimizer updates
    """
    # test incorrect configs
    with pytest.raises(IndexError):
        assert Trainer(accumulate_grad_batches={0: 3, 1: 4, 4: 6})
        assert Trainer(accumulate_grad_batches={-2: 3})

    with pytest.raises(TypeError):
        assert Trainer(accumulate_grad_batches={})
        assert Trainer(accumulate_grad_batches=[[2, 3], [4, 6]])
        assert Trainer(accumulate_grad_batches={1: 2, 3.: 4})
        assert Trainer(accumulate_grad_batches={1: 2.5, 3: 5})

    # test optimizer call freq matches scheduler
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        # only test the first 12 batches in epoch
        if batch_idx < 12:
            if epoch == 0:
                # reset counter when starting epoch
                if batch_idx == 0:
                    self.prev_called_batch_idx = 0

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 1

                assert batch_idx == self.prev_called_batch_idx
                self.prev_called_batch_idx += 1

            elif 1 <= epoch <= 2:
                # reset counter when starting epoch
                if batch_idx == 1:
                    self.prev_called_batch_idx = 1

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 2

                assert batch_idx == self.prev_called_batch_idx
                self.prev_called_batch_idx += 2

            else:
                if batch_idx == 3:
                    self.prev_called_batch_idx = 3

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 4

                assert batch_idx == self.prev_called_batch_idx
                self.prev_called_batch_idx += 3

        optimizer.step()

        # clear gradients
        optimizer.zero_grad()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)
    schedule = {1: 2, 3: 4}

    trainer = Trainer(accumulate_grad_batches=schedule,
                      train_percent_check=0.1,
                      val_percent_check=0.1,
                      max_epochs=4,
                      default_save_path=tmpdir)

    # for the test
    trainer.optimizer_step = optimizer_step
    model.prev_called_batch_idx = 0

    trainer.fit(model)


def test_loading_meta_tags(tmpdir):
    tutils.reset_seed()

    from argparse import Namespace
    hparams = tutils.get_hparams()

    # save tags
    logger = tutils.get_test_tube_logger(tmpdir, False)
    logger.log_hyperparams(Namespace(some_str='a_str', an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load tags
    tags_path = logger.experiment.get_data_path(
        logger.experiment.name, logger.experiment.version
    ) + '/meta_tags.csv'
    tags = training_io.load_hparams_from_tags_csv(tags_path)

    assert tags.batch_size == 32 and tags.hidden_dim == 1000


def test_dp_output_reduce():
    mixin = TrainerLoggingMixin()
    tutils.reset_seed()

    # test identity when we have a single gpu
    out = torch.rand(3, 1)
    assert mixin.reduce_distributed_output(out, num_gpus=1) is out

    # average when we have multiples
    assert mixin.reduce_distributed_output(out, num_gpus=2) == out.mean()

    # when we have a dict of vals
    out = {
        'a': out,
        'b': {
            'c': out
        }
    }
    reduced = mixin.reduce_distributed_output(out, num_gpus=3)
    assert reduced['a'] == out['a']
    assert reduced['b']['c'] == out['b']['c']


def test_model_checkpoint_options(tmp_path):
    """Test ModelCheckpoint options."""
    def mock_save_function(filepath):
        open(filepath, 'a').close()

    hparams = tutils.get_hparams()
    _ = LightningTestModel(hparams)

    # simulated losses
    save_dir = tmp_path / "1"
    save_dir.mkdir()
    losses = [10, 9, 2.8, 5, 2.5]

    # -----------------
    # CASE K=-1  (all)
    w = ModelCheckpoint(save_dir, save_top_k=-1, verbose=1)
    w.save_function = mock_save_function
    for i, loss in enumerate(losses):
        w.on_epoch_end(i, logs={'val_loss': loss})

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == len(losses), "Should save all models when save_top_k=-1"

    # verify correct naming
    for i in range(0, len(losses)):
        assert f'_ckpt_epoch_{i}.ckpt' in file_lists

    save_dir = tmp_path / "2"
    save_dir.mkdir()

    # -----------------
    # CASE K=0 (none)
    w = ModelCheckpoint(save_dir, save_top_k=0, verbose=1)
    w.save_function = mock_save_function
    for i, loss in enumerate(losses):
        w.on_epoch_end(i, logs={'val_loss': loss})

    file_lists = os.listdir(save_dir)

    assert len(file_lists) == 0, "Should save 0 models when save_top_k=0"

    save_dir = tmp_path / "3"
    save_dir.mkdir()

    # -----------------
    # CASE K=1 (2.5, epoch 4)
    w = ModelCheckpoint(save_dir, save_top_k=1, verbose=1, prefix='test_prefix')
    w.save_function = mock_save_function
    for i, loss in enumerate(losses):
        w.on_epoch_end(i, logs={'val_loss': loss})

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == 1, "Should save 1 model when save_top_k=1"
    assert 'test_prefix_ckpt_epoch_4.ckpt' in file_lists

    save_dir = tmp_path / "4"
    save_dir.mkdir()

    # -----------------
    # CASE K=2 (2.5 epoch 4, 2.8 epoch 2)
    # make sure other files don't get deleted

    w = ModelCheckpoint(save_dir, save_top_k=2, verbose=1)
    open(f'{save_dir}/other_file.ckpt', 'a').close()
    w.save_function = mock_save_function
    for i, loss in enumerate(losses):
        w.on_epoch_end(i, logs={'val_loss': loss})

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == 3, 'Should save 2 model when save_top_k=2'
    assert '_ckpt_epoch_4.ckpt' in file_lists
    assert '_ckpt_epoch_2.ckpt' in file_lists
    assert 'other_file.ckpt' in file_lists

    save_dir = tmp_path / "5"
    save_dir.mkdir()

    # -----------------
    # CASE K=4 (save all 4 models)
    # multiple checkpoints within same epoch

    w = ModelCheckpoint(save_dir, save_top_k=4, verbose=1)
    w.save_function = mock_save_function
    for loss in losses:
        w.on_epoch_end(0, logs={'val_loss': loss})

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == 4, 'Should save all 4 models when save_top_k=4 within same epoch'

    save_dir = tmp_path / "6"
    save_dir.mkdir()

    # -----------------
    # CASE K=3 (save the 2nd, 3rd, 4th model)
    # multiple checkpoints within same epoch

    w = ModelCheckpoint(save_dir, save_top_k=3, verbose=1)
    w.save_function = mock_save_function
    for loss in losses:
        w.on_epoch_end(0, logs={'val_loss': loss})

    file_lists = set(os.listdir(save_dir))

    assert len(file_lists) == 3, 'Should save 3 models when save_top_k=3'
    assert '_ckpt_epoch_0_v2.ckpt' in file_lists
    assert '_ckpt_epoch_0_v1.ckpt' in file_lists
    assert '_ckpt_epoch_0.ckpt' in file_lists


def test_model_freeze_unfreeze():
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    model.freeze()
    model.unfreeze()


def test_multiple_val_dataloader(tmpdir):
    """Verify multiple val_dataloader."""
    tutils.reset_seed()

    class CurrentTestModel(
        LightningValidationMultipleDataloadersMixin,
        LightningTestModelBase
    ):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=1.0,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # verify training completed
    assert result == 1

    # verify there are 2 val loaders
    assert len(trainer.get_val_dataloaders()) == 2, \
        'Multiple val_dataloaders not initiated properly'

    # make sure predictions are good for each val set
    for dataloader in trainer.get_val_dataloaders():
        tutils.run_prediction(dataloader, trainer.model)


def test_multiple_test_dataloader(tmpdir):
    """Verify multiple test_dataloader."""
    tutils.reset_seed()

    class CurrentTestModel(
        LightningTestMultipleDataloadersMixin,
        LightningTestModelBase
    ):
        pass

    hparams = tutils.get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        default_save_path=tmpdir,
        max_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.2,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # verify there are 2 val loaders
    assert len(trainer.get_test_dataloaders()) == 2, \
        'Multiple test_dataloaders not initiated properly'

    # make sure predictions are good for each test set
    for dataloader in trainer.get_test_dataloaders():
        tutils.run_prediction(dataloader, trainer.model)

    # run the test method
    trainer.test()


# if __name__ == '__main__':
#     pytest.main([__file__])
