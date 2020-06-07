import glob
import logging as log
import os
import pickle

import cloudpickle
import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from tests.base import EvalModelTemplate


@pytest.mark.spawn
@pytest.mark.parametrize("backend", ['dp', 'ddp'])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_running_test_pretrained_model_distrib(tmpdir, backend):
    """Verify `test()` on pretrained model."""
    tutils.set_random_master_port()

    model = EvalModelTemplate()

    # exp file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        train_percent_check=0.4,
        val_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger,
        gpus=[0, 1],
        distributed_backend=backend,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    log.info(os.listdir(tutils.get_data_path(logger, path_dir=tmpdir)))

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = tutils.load_model(logger,
                                         trainer.checkpoint_callback.dirpath,
                                         module_class=EvalModelTemplate)

    # run test set
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    tutils.assert_ok_model_acc(new_trainer)

    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        tutils.run_prediction(dataloader, pretrained_model)


def test_running_test_pretrained_model_cpu(tmpdir):
    """Verify test() on pretrained model."""
    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=3,
        train_percent_check=0.4,
        val_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = tutils.load_model(
        logger, trainer.checkpoint_callback.dirpath, module_class=EvalModelTemplate
    )

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    tutils.assert_ok_model_acc(new_trainer)


def test_load_model_from_checkpoint(tmpdir):
    """Verify test() on pretrained model."""
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer_options = dict(
        progress_bar_refresh_rate=0,
        max_epochs=2,
        train_percent_check=0.4,
        val_percent_check=0.2,
        checkpoint_callback=ModelCheckpoint(tmpdir, save_top_k=-1),
        default_root_dir=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    trainer.test()

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'

    # load last checkpoint
    last_checkpoint = sorted(glob.glob(os.path.join(trainer.checkpoint_callback.dirpath, "*.ckpt")))[-1]
    pretrained_model = EvalModelTemplate.load_from_checkpoint(last_checkpoint)

    # test that hparams loaded correctly
    for k, v in hparams.items():
        assert getattr(pretrained_model, k) == v

    # assert weights are the same
    for (old_name, old_p), (new_name, new_p) in zip(model.named_parameters(), pretrained_model.named_parameters()):
        assert torch.all(torch.eq(old_p, new_p)), 'loaded weights are not the same as the saved weights'

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    tutils.assert_ok_model_acc(new_trainer)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_dp_resume(tmpdir):
    """Make sure DP continues training correctly."""
    hparams = EvalModelTemplate.get_default_hparams()
    model = EvalModelTemplate(**hparams)

    trainer_options = dict(
        max_epochs=1,
        gpus=2,
        distributed_backend='dp',
    )

    # get logger
    logger = tutils.get_default_logger(tmpdir)

    # exp file to get weights
    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    # add these to the trainer options
    trainer_options['logger'] = logger
    trainer_options['checkpoint_callback'] = checkpoint

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.is_slurm_managing_tasks = True
    result = trainer.fit(model)

    # track epoch before saving. Increment since we finished the current epoch, don't want to rerun
    real_global_epoch = trainer.current_epoch + 1

    # correct result and ok accuracy
    assert result == 1, 'amp + dp model failed to complete'

    # ---------------------------
    # HPC LOAD/SAVE
    # ---------------------------
    # save
    trainer.hpc_save(tmpdir, logger)

    # init new trainer
    new_logger = tutils.get_default_logger(tmpdir, version=logger.version)
    trainer_options['logger'] = new_logger
    trainer_options['checkpoint_callback'] = ModelCheckpoint(tmpdir)
    trainer_options['train_percent_check'] = 0.5
    trainer_options['val_percent_check'] = 0.2
    trainer_options['max_epochs'] = 1
    new_trainer = Trainer(**trainer_options)

    # set the epoch start hook so we can predict before the model does the full training
    def assert_good_acc():
        assert new_trainer.current_epoch == real_global_epoch and new_trainer.current_epoch > 0

        # if model and state loaded correctly, predictions will be good even though we
        # haven't trained with the new loaded model
        dp_model = new_trainer.model
        dp_model.eval()

        dataloader = trainer.train_dataloader
        tutils.run_prediction(dataloader, dp_model, dp=True)

    # new model
    model = EvalModelTemplate(**hparams)
    model.on_train_start = assert_good_acc

    # fit new model which should load hpc weights
    new_trainer.fit(model)

    # test freeze on gpu
    model.freeze()
    model.unfreeze()


def test_model_saving_loading(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    model = EvalModelTemplate()

    # logger file to get meta
    logger = tutils.get_default_logger(tmpdir)

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

    # make a prediction
    dataloaders = model.test_dataloader()
    if not isinstance(dataloaders, list):
        dataloaders = [dataloaders]

    for dataloader in dataloaders:
        for batch in dataloader:
            break

    x, y = batch
    x = x.view(x.size(0), -1)

    # generate preds before saving model
    model.eval()
    pred_before_saving = model(x)

    # save model
    new_weights_path = os.path.join(tmpdir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    hparams_path = tutils.get_data_path(logger, path_dir=tmpdir)
    hparams_path = os.path.join(hparams_path, 'hparams.yaml')
    model_2 = EvalModelTemplate.load_from_checkpoint(
        checkpoint_path=new_weights_path,
        hparams_file=hparams_path
    )
    model_2.eval()

    # make prediction
    # assert that both predictions are the same
    new_pred = model_2(x)
    assert torch.all(torch.eq(pred_before_saving, new_pred)).item() == 1


def test_model_pickle(tmpdir):
    model = EvalModelTemplate()
    pickle.dumps(model)
    cloudpickle.dumps(model)
