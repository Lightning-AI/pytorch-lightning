import logging
import os

import torch

import tests.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.testing import LightningTestModel


def test_running_test_pretrained_model_ddp(tmpdir):
    """Verify `test()` on pretrained model."""
    if not tutils.can_run_gpu_test():
        return

    tutils.reset_seed()
    tutils.set_random_master_port()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    # exp file to get meta
    logger = tutils.get_test_tube_logger(tmpdir, False)

    # exp file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=False,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger,
        gpus=[0, 1],
        distributed_backend='ddp'
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    exp = logger.experiment
    logging.info(os.listdir(exp.get_data_path(exp.name, exp.version)))

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = tutils.load_model(logger.experiment,
                                         trainer.checkpoint_callback.filepath,
                                         module_class=LightningTestModel)

    # run test set
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    for dataloader in model.test_dataloader():
        tutils.run_prediction(dataloader, pretrained_model)


def test_running_test_pretrained_model(tmpdir):
    tutils.reset_seed()

    """Verify test() on pretrained model"""
    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    # logger file to get meta
    logger = tutils.get_test_tube_logger(tmpdir, False)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=False,
        max_epochs=4,
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
        logger.experiment, trainer.checkpoint_callback.filepath, module_class=LightningTestModel
    )

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    tutils.assert_ok_test_acc(new_trainer)


def test_load_model_from_checkpoint(tmpdir):
    tutils.reset_seed()

    """Verify test() on pretrained model"""
    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        show_progress_bar=False,
        max_epochs=2,
        train_percent_check=0.4,
        val_percent_check=0.2,
        checkpoint_callback=ModelCheckpoint(tmpdir, save_top_k=-1),
        logger=False,
        default_save_path=tmpdir,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'

    # load last checkpoint
    last_checkpoint = os.path.join(trainer.checkpoint_callback.filepath, "_ckpt_epoch_1.ckpt")
    if not os.path.isfile(last_checkpoint):
        last_checkpoint = os.path.join(trainer.checkpoint_callback.filepath, "_ckpt_epoch_0.ckpt")
    pretrained_model = LightningTestModel.load_from_checkpoint(last_checkpoint)

    # test that hparams loaded correctly
    for k, v in vars(hparams).items():
        assert getattr(pretrained_model.hparams, k) == v

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    tutils.assert_ok_test_acc(new_trainer)


def test_running_test_pretrained_model_dp(tmpdir):
    tutils.reset_seed()

    """Verify test() on pretrained model"""
    if not tutils.can_run_gpu_test():
        return

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    # logger file to get meta
    logger = tutils.get_test_tube_logger(tmpdir, False)

    # logger file to get weights
    checkpoint = tutils.init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=True,
        max_epochs=4,
        train_percent_check=0.4,
        val_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger,
        gpus=[0, 1],
        distributed_backend='dp'
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = tutils.load_model(logger.experiment,
                                         trainer.checkpoint_callback.filepath,
                                         module_class=LightningTestModel)

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    tutils.assert_ok_test_acc(new_trainer)


def test_dp_resume(tmpdir):
    """Make sure DP continues training correctly."""
    if not tutils.can_run_gpu_test():
        return

    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        show_progress_bar=True,
        max_epochs=2,
        gpus=2,
        distributed_backend='dp',
    )

    # get logger
    logger = tutils.get_test_tube_logger(tmpdir, debug=False)

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

    # track epoch before saving
    real_global_epoch = trainer.current_epoch

    # correct result and ok accuracy
    assert result == 1, 'amp + dp model failed to complete'

    # ---------------------------
    # HPC LOAD/SAVE
    # ---------------------------
    # save
    trainer.hpc_save(tmpdir, logger)

    # init new trainer
    new_logger = tutils.get_test_tube_logger(tmpdir, version=logger.version)
    trainer_options['logger'] = new_logger
    trainer_options['checkpoint_callback'] = ModelCheckpoint(tmpdir)
    trainer_options['train_percent_check'] = 0.2
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

        dataloader = trainer.get_train_dataloader()
        tutils.run_prediction(dataloader, dp_model, dp=True)

    # new model
    model = LightningTestModel(hparams)
    model.on_train_start = assert_good_acc

    # fit new model which should load hpc weights
    new_trainer.fit(model)

    # test freeze on gpu
    model.freeze()
    model.unfreeze()


def test_cpu_restore_training(tmpdir):
    """Verify continue training session on CPU."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

    # logger file to get meta
    test_logger_version = 10
    logger = tutils.get_test_tube_logger(tmpdir, False, version=test_logger_version)

    trainer_options = dict(
        max_epochs=8,
        val_check_interval=0.50,
        val_percent_check=0.2,
        train_percent_check=0.2,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(tmpdir, save_top_k=-1)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    real_global_epoch = trainer.current_epoch

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    # wipe-out trainer and model
    # retrain with not much data... this simulates picking training back up after slurm
    # we want to see if the weights come back correctly
    new_logger = tutils.get_test_tube_logger(tmpdir, False, version=test_logger_version)
    trainer_options = dict(
        max_epochs=2,
        val_check_interval=0.50,
        val_percent_check=0.2,
        train_percent_check=0.2,
        logger=new_logger,
        checkpoint_callback=ModelCheckpoint(tmpdir),
    )
    trainer = Trainer(**trainer_options)
    model = LightningTestModel(hparams)

    # set the epoch start hook so we can predict before the model does the full training
    def assert_good_acc():
        assert trainer.current_epoch == real_global_epoch
        assert trainer.current_epoch >= 0

        # if model and state loaded correctly, predictions will be good even though we
        # haven't trained with the new loaded model
        trainer.model.eval()
        for dataloader in trainer.get_val_dataloaders():
            tutils.run_prediction(dataloader, trainer.model)

    model.on_train_start = assert_good_acc

    # by calling fit again, we trigger training, loading weights from the cluster
    # and our hook to predict using current model before any more weight updates
    trainer.fit(model)


def test_model_saving_loading(tmpdir):
    """Tests use case where trainer saves the model, and user loads it from tags independently."""
    tutils.reset_seed()

    hparams = tutils.get_hparams()
    model = LightningTestModel(hparams)

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

    # make a prediction
    for dataloader in model.test_dataloader():
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
    tags_path = logger.experiment.get_data_path(logger.experiment.name, logger.experiment.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_metrics(weights_path=new_weights_path,
                                                   tags_csv=tags_path)
    model_2.eval()

    # make prediction
    # assert that both predictions are the same
    new_pred = model_2(x)
    assert torch.all(torch.eq(pred_before_saving, new_pred)).item() == 1

# if __name__ == '__main__':
#     pytest.main([__file__])
