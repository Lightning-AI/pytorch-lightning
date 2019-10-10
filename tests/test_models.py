import os
import shutil
import warnings
from argparse import Namespace

import pytest
import numpy as np
import torch

# sys.path += [os.path.abspath('..'), os.path.abspath('../..')]
from pytorch_lightning import Trainer
from pytorch_lightning.testing import (
    LightningTestModel,
    LightningTestModelBase,
    LightningValidationMixin,
    LightningValidationStepMixin,
    LightningValidationMultipleDataloadersMixin,
    LightningTestMixin,
    LightningTestMultipleDataloadersMixin,
)
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    GradientAccumulationScheduler,
)
from pytorch_lightning.utilities.debugging import MisconfigurationException
from pytorch_lightning.root_module import memory
from pytorch_lightning.trainer.trainer import reduce_distributed_output
from pytorch_lightning.root_module import model_saving
from pytorch_lightning.trainer import trainer_io
from pytorch_lightning.logging import TestTubeLogger
from examples import LightningTemplateModel

# generate a list of random seeds for each test
RANDOM_PORTS = list(np.random.randint(12000, 19000, 1000))
ROOT_SEED = 1234
torch.manual_seed(ROOT_SEED)
np.random.seed(ROOT_SEED)
RANDOM_SEEDS = list(np.random.randint(0, 10000, 1000))


# ------------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------------
def test_running_test_pretrained_model_ddp():
    """Verify test() on pretrained model"""
    if not can_run_gpu_test():
        return

    reset_seed()
    set_random_master_port()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
    logger = get_test_tube_logger(False)

    # exp file to get weights
    checkpoint = init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
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
    print(os.listdir(exp.get_data_path(exp.name, exp.version)))

    # correct result and ok accuracy
    assert result == 1, 'training failed to complete'
    pretrained_model = load_model(logger.experiment, save_dir,
                                  module_class=LightningTestModel)

    # run test set
    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    [run_prediction(dataloader, pretrained_model) for dataloader in model.test_dataloader()]

    # test we have good test accuracy
    clear_save_dir()


def test_default_logger_callbacks_cpu_model():
    """
    Test each of the trainer options
    :return:
    """
    reset_seed()

    trainer_options = dict(
        max_nb_epochs=1,
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        print_nan_grads=True,
        show_progress_bar=False,
        train_percent_check=0.01,
        val_percent_check=0.01
    )

    model, hparams = get_model()
    run_model_test_no_loggers(trainer_options, model, hparams, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


def test_lbfgs_cpu_model():
    """
    Test each of the trainer options
    :return:
    """
    reset_seed()

    trainer_options = dict(
        max_nb_epochs=1,
        gradient_clip_val=1.0,
        print_nan_grads=True,
        show_progress_bar=False,
        weights_summary='top',
        train_percent_check=1.0,
        val_percent_check=0.2
    )

    model, hparams = get_model(use_test_model=True, lbfgs=True)
    run_model_test_no_loggers(trainer_options, model, hparams, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


def test_multi_gpu_model_ddp2():
    """
    Make sure DDP2 works
    :return:
    """
    if not can_run_gpu_test():
        return

    reset_seed()
    set_random_master_port()

    model, hparams = get_model()
    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        gpus=2,
        weights_summary=None,
        distributed_backend='ddp2'
    )

    run_gpu_model_test(trainer_options, model, hparams)


def test_dp_resume():
    """
    Make sure DP continues training correctly
    :return:
    """
    if not can_run_gpu_test():
        return

    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=2,
        gpus=2,
        distributed_backend='dp',
    )

    save_dir = init_save_dir()

    # get logger
    logger = get_test_tube_logger(debug=False)

    # exp file to get weights
    # logger file to get weights
    checkpoint = init_checkpoint_callback(logger)

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
    trainer.hpc_save(save_dir, logger)

    # init new trainer
    new_logger = get_test_tube_logger(version=logger.version)
    trainer_options['logger'] = new_logger
    trainer_options['checkpoint_callback'] = ModelCheckpoint(save_dir)
    trainer_options['train_percent_check'] = 0.2
    trainer_options['val_percent_check'] = 0.2
    trainer_options['max_nb_epochs'] = 1
    new_trainer = Trainer(**trainer_options)

    # set the epoch start hook so we can predict before the model does the full training
    def assert_good_acc():
        assert new_trainer.current_epoch == real_global_epoch and new_trainer.current_epoch > 0

        # if model and state loaded correctly, predictions will be good even though we
        # haven't trained with the new loaded model
        dp_model = new_trainer.model
        dp_model.eval()

        dataloader = trainer.get_train_dataloader()
        run_prediction(dataloader, dp_model, dp=True)

    # new model
    model = LightningTestModel(hparams)
    model.on_sanity_check_start = assert_good_acc

    # fit new model which should load hpc weights
    new_trainer.fit(model)

    # test freeze on gpu
    model.freeze()
    model.unfreeze()

    clear_save_dir()


def test_running_test_after_fitting():
    """Verify test() on fitted model"""
    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    # logger file to get weights
    checkpoint = init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        test_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'

    trainer.test()

    # test we have good test accuracy
    assert_ok_test_acc(trainer)

    clear_save_dir()


def test_running_test_without_val():
    reset_seed()

    """Verify test() works on a model with no val_loader"""
    class CurrentTestModel(LightningTestMixin, LightningTestModelBase):
        pass
    hparams = get_hparams()
    model = CurrentTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    # logger file to get weights
    checkpoint = init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        test_percent_check=0.2,
        checkpoint_callback=checkpoint,
        logger=logger
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    assert result == 1, 'training failed to complete'

    trainer.test()

    # test we have good test accuracy
    assert_ok_test_acc(trainer)

    clear_save_dir()


def test_running_test_pretrained_model():
    reset_seed()

    """Verify test() on pretrained model"""
    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    # logger file to get weights
    checkpoint = init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
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
    pretrained_model = load_model(
        logger.experiment, save_dir, module_class=LightningTestModel
    )

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    assert_ok_test_acc(new_trainer)
    clear_save_dir()


def test_running_test_pretrained_model_dp():
    reset_seed()

    """Verify test() on pretrained model"""
    if not can_run_gpu_test():
        return

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    # logger file to get weights
    checkpoint = init_checkpoint_callback(logger)

    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=1,
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
    pretrained_model = load_model(logger.experiment, save_dir,
                                  module_class=LightningTestModel)

    new_trainer = Trainer(**trainer_options)
    new_trainer.test(pretrained_model)

    # test we have good test accuracy
    assert_ok_test_acc(new_trainer)
    clear_save_dir()


def test_gradient_accumulation_scheduling():
    reset_seed()

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
    def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
        # only test the first 12 batches in epoch
        if batch_nb < 12:
            if epoch_nb == 0:
                # reset counter when starting epoch
                if batch_nb == 0:
                    self.prev_called_batch_nb = 0

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 1

                assert batch_nb == self.prev_called_batch_nb
                self.prev_called_batch_nb += 1

            elif 1 <= epoch_nb <= 2:
                # reset counter when starting epoch
                if batch_nb == 1:
                    self.prev_called_batch_nb = 1

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 2

                assert batch_nb == self.prev_called_batch_nb
                self.prev_called_batch_nb += 2

            else:
                if batch_nb == 3:
                    self.prev_called_batch_nb = 3

                    # use this opportunity to test once
                    assert self.trainer.accumulate_grad_batches == 4

                assert batch_nb == self.prev_called_batch_nb
                self.prev_called_batch_nb += 3

        optimizer.step()

        # clear gradients
        optimizer.zero_grad()

    hparams = get_hparams()
    model = LightningTestModel(hparams)
    schedule = {1: 2, 3: 4}

    trainer = Trainer(accumulate_grad_batches=schedule,
                      train_percent_check=0.1,
                      val_percent_check=0.1,
                      max_nb_epochs=4)

    # for the test
    trainer.optimizer_step = optimizer_step
    model.prev_called_batch_nb = 0

    trainer.fit(model)


def test_multi_gpu_model_ddp():
    """
    Make sure DDP works
    :return:
    """
    if not can_run_gpu_test():
        return

    reset_seed()
    set_random_master_port()

    model, hparams = get_model()
    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        gpus=[0, 1],
        distributed_backend='ddp'
    )

    run_gpu_model_test(trainer_options, model, hparams)


def test_optimizer_return_options():
    reset_seed()

    trainer = Trainer()
    model, hparams = get_model()

    # single optimizer
    opt_a = torch.optim.Adam(model.parameters(), lr=0.002)
    opt_b = torch.optim.SGD(model.parameters(), lr=0.002)
    optim, lr_sched = trainer.init_optimizers(opt_a)
    assert len(optim) == 1 and len(lr_sched) == 0

    # opt tuple
    opts = (opt_a, opt_b)
    optim, lr_sched = trainer.init_optimizers(opts)
    assert len(optim) == 2 and optim[0] == opts[0] and optim[1] == opts[1]
    assert len(lr_sched) == 0

    # opt list
    opts = [opt_a, opt_b]
    optim, lr_sched = trainer.init_optimizers(opts)
    assert len(optim) == 2 and optim[0] == opts[0] and optim[1] == opts[1]
    assert len(lr_sched) == 0

    # opt tuple of lists
    opts = ([opt_a], ['lr_scheduler'])
    optim, lr_sched = trainer.init_optimizers(opts)
    assert len(optim) == 1 and len(lr_sched) == 1
    assert optim[0] == opts[0][0] and lr_sched[0] == 'lr_scheduler'


def test_single_gpu_batch_parse():
    reset_seed()

    if not can_run_gpu_test():
        return

    trainer = Trainer()

    # batch is just a tensor
    batch = torch.rand(2, 3)
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch.device.index == 0 and batch.type() == 'torch.cuda.FloatTensor'

    # tensor list
    batch = [torch.rand(2, 3), torch.rand(2, 3)]
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0].device.index == 0 and batch[0].type() == 'torch.cuda.FloatTensor'
    assert batch[1].device.index == 0 and batch[1].type() == 'torch.cuda.FloatTensor'

    # tensor list of lists
    batch = [[torch.rand(2, 3), torch.rand(2, 3)]]
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0][0].device.index == 0 and batch[0][0].type() == 'torch.cuda.FloatTensor'
    assert batch[0][1].device.index == 0 and batch[0][1].type() == 'torch.cuda.FloatTensor'

    # tensor dict
    batch = [{'a': torch.rand(2, 3), 'b': torch.rand(2, 3)}]
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0]['a'].device.index == 0 and batch[0]['a'].type() == 'torch.cuda.FloatTensor'
    assert batch[0]['b'].device.index == 0 and batch[0]['b'].type() == 'torch.cuda.FloatTensor'

    # tuple of tensor list and list of tensor dict
    batch = ([torch.rand(2, 3) for _ in range(2)],
             [{'a': torch.rand(2, 3), 'b': torch.rand(2, 3)} for _ in range(2)])
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0][0].device.index == 0 and batch[0][0].type() == 'torch.cuda.FloatTensor'

    assert batch[1][0]['a'].device.index == 0
    assert batch[1][0]['a'].type() == 'torch.cuda.FloatTensor'

    assert batch[1][0]['b'].device.index == 0
    assert batch[1][0]['b'].type() == 'torch.cuda.FloatTensor'


def test_early_stopping_cpu_model():
    """
    Test each of the trainer options
    :return:
    """
    reset_seed()

    stopping = EarlyStopping(monitor='val_loss')
    trainer_options = dict(
        early_stop_callback=stopping,
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        track_grad_norm=2,
        print_nan_grads=True,
        show_progress_bar=True,
        logger=get_test_tube_logger(),
        train_percent_check=0.1,
        val_percent_check=0.1
    )

    model, hparams = get_model()
    run_gpu_model_test(trainer_options, model, hparams, on_gpu=False)

    # test freeze on cpu
    model.freeze()
    model.unfreeze()


def test_no_val_module():
    """
    Tests use case where trainer saves the model, and user loads it from tags independently
    :return:
    """
    reset_seed()

    hparams = get_hparams()

    class CurrentTestModel(LightningTestModelBase):
        pass
    model = CurrentTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    trainer_options = dict(
        max_nb_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(save_dir)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # training complete
    assert result == 1, 'amp + ddp model failed to complete'

    # save model
    new_weights_path = os.path.join(save_dir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    tags_path = logger.experiment.get_data_path(logger.experiment.name, logger.experiment.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_metrics(weights_path=new_weights_path,
                                                   tags_csv=tags_path)
    model_2.eval()

    # make prediction
    clear_save_dir()


def test_no_val_end_module():
    """
    Tests use case where trainer saves the model, and user loads it from tags independently
    :return:
    """
    reset_seed()

    class CurrentTestModel(LightningValidationStepMixin, LightningTestModelBase):
        pass
    hparams = get_hparams()
    model = CurrentTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    trainer_options = dict(
        max_nb_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(save_dir)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    # save model
    new_weights_path = os.path.join(save_dir, 'save_test.ckpt')
    trainer.save_checkpoint(new_weights_path)

    # load new model
    tags_path = logger.experiment.get_data_path(logger.experiment.name, logger.experiment.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_metrics(weights_path=new_weights_path,
                                                   tags_csv=tags_path)
    model_2.eval()

    # make prediction
    clear_save_dir()


def test_simple_cpu():
    """
    Verify continue training session on CPU
    :return:
    """
    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    trainer_options = dict(
        max_nb_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.1,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    clear_save_dir()


def test_amp_single_gpu():
    """
    Make sure DDP + AMP work
    :return:
    """
    reset_seed()

    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      'Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      'Rerun on a node with 2+ GPUs to run this test')
        return

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=1,
        gpus=1,
        distributed_backend='ddp',
        use_amp=True
    )

    run_gpu_model_test(trainer_options, model, hparams)


def test_no_amp_single_gpu():
    """
    Make sure DDP + AMP work
    :return:
    """
    reset_seed()

    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      'Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      'Rerun on a node with 2+ GPUs to run this test')
        return

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=1,
        gpus=1,
        distributed_backend='dp',
        use_amp=True
    )

    with pytest.raises((MisconfigurationException, ModuleNotFoundError)):
        run_gpu_model_test(trainer_options, model, hparams)


def test_cpu_restore_training():
    """
    Verify continue training session on CPU
    :return:
    """
    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    test_logger_version = 10
    logger = get_test_tube_logger(False, version=test_logger_version)

    trainer_options = dict(
        max_nb_epochs=2,
        val_check_interval=0.50,
        val_percent_check=0.2,
        train_percent_check=0.2,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(save_dir)
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
    new_logger = get_test_tube_logger(False, version=test_logger_version)
    trainer_options = dict(
        max_nb_epochs=2,
        val_check_interval=0.50,
        val_percent_check=0.2,
        train_percent_check=0.2,
        logger=new_logger,
        checkpoint_callback=ModelCheckpoint(save_dir),
    )
    trainer = Trainer(**trainer_options)
    model = LightningTestModel(hparams)

    # set the epoch start hook so we can predict before the model does the full training
    def assert_good_acc():
        assert trainer.current_epoch == real_global_epoch and trainer.current_epoch > 0

        # if model and state loaded correctly, predictions will be good even though we
        # haven't trained with the new loaded model
        trainer.model.eval()
        for dataloader in trainer.get_val_dataloaders():
            run_prediction(dataloader, trainer.model)

    model.on_sanity_check_start = assert_good_acc

    # by calling fit again, we trigger training, loading weights from the cluster
    # and our hook to predict using current model before any more weight updates
    trainer.fit(model)

    clear_save_dir()


def test_amp_gpu_ddp():
    """
    Make sure DDP + AMP work
    :return:
    """
    if not can_run_gpu_test():
        return

    reset_seed()
    set_random_master_port()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=1,
        gpus=2,
        distributed_backend='ddp',
        use_amp=True
    )

    run_gpu_model_test(trainer_options, model, hparams)


def test_cpu_slurm_save_load():
    """
    Verify model save/load/checkpoint on CPU
    :return:
    """
    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    version = logger.version

    trainer_options = dict(
        max_nb_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(save_dir)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    real_global_step = trainer.global_step

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    # predict with trained model before saving
    # make a prediction
    for dataloader in model.test_dataloader():
        for batch in dataloader:
            break

    x, y = batch
    x = x.view(x.size(0), -1)

    model.eval()
    pred_before_saving = model(x)

    # test HPC saving
    # simulate snapshot on slurm
    saved_filepath = trainer.hpc_save(save_dir, logger)
    assert os.path.exists(saved_filepath)

    # new logger file to get meta
    logger = get_test_tube_logger(False, version=version)

    trainer_options = dict(
        max_nb_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(save_dir),
    )
    trainer = Trainer(**trainer_options)
    model = LightningTestModel(hparams)

    # set the epoch start hook so we can predict before the model does the full training
    def assert_pred_same():
        assert trainer.global_step == real_global_step and trainer.global_step > 0

        # predict with loaded model to make sure answers are the same
        trainer.model.eval()
        new_pred = trainer.model(x)
        assert torch.all(torch.eq(pred_before_saving, new_pred)).item() == 1

    model.on_epoch_start = assert_pred_same

    # by calling fit again, we trigger training, loading weights from the cluster
    # and our hook to predict using current model before any more weight updates
    trainer.fit(model)

    clear_save_dir()


def test_loading_meta_tags():
    reset_seed()

    from argparse import Namespace
    hparams = get_hparams()

    # save tags
    logger = get_test_tube_logger(False)
    logger.log_hyperparams(Namespace(some_str='a_str', an_int=1, a_float=2.0))
    logger.log_hyperparams(hparams)
    logger.save()

    # load tags
    tags_path = logger.experiment.get_data_path(
        logger.experiment.name, logger.experiment.version
    ) + '/meta_tags.csv'
    tags = trainer_io.load_hparams_from_tags_csv(tags_path)

    assert tags.batch_size == 32 and tags.hidden_dim == 1000

    clear_save_dir()


def test_dp_output_reduce():
    reset_seed()

    # test identity when we have a single gpu
    out = torch.rand(3, 1)
    assert reduce_distributed_output(out, nb_gpus=1) is out

    # average when we have multiples
    assert reduce_distributed_output(out, nb_gpus=2) == out.mean()

    # when we have a dict of vals
    out = {
        'a': out,
        'b': {
            'c': out
        }
    }
    reduced = reduce_distributed_output(out, nb_gpus=3)
    assert reduced['a'] == out['a']
    assert reduced['b']['c'] == out['b']['c']


def test_model_saving_loading():
    """
    Tests use case where trainer saves the model, and user loads it from tags independently
    :return:
    """
    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    trainer_options = dict(
        max_nb_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(save_dir)
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
    new_weights_path = os.path.join(save_dir, 'save_test.ckpt')
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

    clear_save_dir()


def test_model_freeze_unfreeze():
    reset_seed()

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    model.freeze()
    model.unfreeze()


def test_amp_gpu_ddp_slurm_managed():
    """
    Make sure DDP + AMP work
    :return:
    """
    if not can_run_gpu_test():
        return

    reset_seed()

    # simulate setting slurm flags
    set_random_master_port()
    os.environ['SLURM_LOCALID'] = str(0)

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=1,
        gpus=[0],
        distributed_backend='ddp',
        use_amp=True
    )

    save_dir = init_save_dir()

    # exp file to get meta
    logger = get_test_tube_logger(False)

    # exp file to get weights
    checkpoint = init_checkpoint_callback(logger)

    # add these to the trainer options
    trainer_options['checkpoint_callback'] = checkpoint
    trainer_options['logger'] = logger

    # fit model
    trainer = Trainer(**trainer_options)
    trainer.is_slurm_managing_tasks = True
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test root model address
    assert trainer.resolve_root_node_address('abc') == 'abc'
    assert trainer.resolve_root_node_address('abc[23]') == 'abc23'
    assert trainer.resolve_root_node_address('abc[23-24]') == 'abc23'
    assert trainer.resolve_root_node_address('abc[23-24, 45-40, 40]') == 'abc23'

    # test model loading with a map_location
    pretrained_model = load_model(logger.experiment, save_dir)

    # test model preds
    [run_prediction(dataloader, pretrained_model) for dataloader in trainer.get_test_dataloaders()]

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()

    # test HPC loading / saving
    trainer.hpc_save(save_dir, logger)
    trainer.hpc_load(save_dir, on_gpu=True)

    # test freeze on gpu
    model.freeze()
    model.unfreeze()

    clear_save_dir()


def test_cpu_model_with_amp():
    """
    Make sure model trains on CPU
    :return:
    """
    reset_seed()

    trainer_options = dict(
        show_progress_bar=False,
        logger=get_test_tube_logger(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4,
        use_amp=True
    )

    model, hparams = get_model()

    with pytest.raises((MisconfigurationException, ModuleNotFoundError)):
        run_gpu_model_test(trainer_options, model, hparams, on_gpu=False)


def test_cpu_model():
    """
    Make sure model trains on CPU
    :return:
    """
    reset_seed()

    trainer_options = dict(
        show_progress_bar=False,
        logger=get_test_tube_logger(),
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4
    )

    model, hparams = get_model()

    run_gpu_model_test(trainer_options, model, hparams, on_gpu=False)


def test_all_features_cpu_model():
    """
    Test each of the trainer options
    :return:
    """
    reset_seed()

    trainer_options = dict(
        gradient_clip_val=1.0,
        overfit_pct=0.20,
        track_grad_norm=2,
        print_nan_grads=True,
        show_progress_bar=False,
        logger=get_test_tube_logger(),
        accumulate_grad_batches=2,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.4
    )

    model, hparams = get_model()
    run_gpu_model_test(trainer_options, model, hparams, on_gpu=False)


def test_single_gpu_model():
    """
    Make sure single GPU works (DP mode)
    :return:
    """
    reset_seed()

    if not torch.cuda.is_available():
        warnings.warn('test_single_gpu_model cannot run.'
                      ' Rerun on a GPU node to run this test')
        return
    model, hparams = get_model()

    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus=1
    )

    run_gpu_model_test(trainer_options, model, hparams)


def test_multi_gpu_none_backend():
    """
    Make sure when using multiple GPUs the user can't use
    distributed_backend = None
    :return:
    """
    reset_seed()

    if not can_run_gpu_test():
        return

    model, hparams = get_model()
    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus='-1'
    )

    with pytest.raises(MisconfigurationException):
        run_gpu_model_test(trainer_options, model, hparams)


def test_multi_gpu_model_dp():
    """
    Make sure DP works
    :return:
    """
    reset_seed()

    if not can_run_gpu_test():
        return

    model, hparams = get_model()
    trainer_options = dict(
        show_progress_bar=False,
        distributed_backend='dp',
        max_nb_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus='-1'
    )

    run_gpu_model_test(trainer_options, model, hparams)

    # test memory helper functions
    memory.get_gpu_memory_map()


def test_amp_gpu_dp():
    """
    Make sure DP + AMP work
    :return:
    """
    reset_seed()

    if not can_run_gpu_test():
        return

    model, hparams = get_model()
    trainer_options = dict(
        max_nb_epochs=1,
        gpus='0, 1',  # test init with gpu string
        distributed_backend='dp',
        use_amp=True
    )
    with pytest.raises(MisconfigurationException):
        run_gpu_model_test(trainer_options, model, hparams)


def test_ddp_sampler_error():
    """
    Make sure DDP + AMP work
    :return:
    """
    if not can_run_gpu_test():
        return

    reset_seed()
    set_random_master_port()

    hparams = get_hparams()
    model = LightningTestModel(hparams, force_remove_distributed_sampler=True)

    logger = get_test_tube_logger(True)

    trainer = Trainer(
        logger=logger,
        show_progress_bar=False,
        max_nb_epochs=1,
        gpus=[0, 1],
        distributed_backend='ddp',
        use_amp=True
    )

    with pytest.warns(UserWarning):
        trainer.get_dataloaders(model)

    clear_save_dir()


def test_multiple_val_dataloader():
    """
    Verify multiple val_dataloader
    :return:
    """
    reset_seed()

    class CurrentTestModel(
        LightningValidationMultipleDataloadersMixin,
        LightningTestModelBase
    ):
        pass
    hparams = get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        max_nb_epochs=1,
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
    [run_prediction(dataloader, trainer.model) for dataloader in trainer.get_val_dataloaders()]


def test_multiple_test_dataloader():
    """
    Verify multiple test_dataloader
    :return:
    """
    reset_seed()

    class CurrentTestModel(
        LightningTestMultipleDataloadersMixin,
        LightningTestModelBase
    ):
        pass
    hparams = get_hparams()
    model = CurrentTestModel(hparams)

    # logger file to get meta
    trainer_options = dict(
        max_nb_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.1,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # verify there are 2 val loaders
    assert len(trainer.get_test_dataloaders()) == 2, \
        'Multiple test_dataloaders not initiated properly'

    # make sure predictions are good for each test set
    [run_prediction(dataloader, trainer.model) for dataloader in trainer.get_test_dataloaders()]

    # run the test method
    trainer.test()


# ------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------
def run_model_test_no_loggers(trainer_options, model, hparams, on_gpu=True):
    save_dir = init_save_dir()

    trainer_options['default_save_path'] = save_dir

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test model loading
    pretrained_model = load_model(trainer.logger.experiment, save_dir)

    # test new model accuracy
    [run_prediction(dataloader, pretrained_model) for dataloader in model.test_dataloader()]

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()

    clear_save_dir()


def run_gpu_model_test(trainer_options, model, hparams, on_gpu=True):
    save_dir = init_save_dir()

    # logger file to get meta
    logger = get_test_tube_logger(False)

    # logger file to get weights
    checkpoint = init_checkpoint_callback(logger)

    # add these to the trainer options
    trainer_options['checkpoint_callback'] = checkpoint
    trainer_options['logger'] = logger

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test model loading
    pretrained_model = load_model(logger.experiment, save_dir)

    # test new model accuracy
    [run_prediction(dataloader, pretrained_model) for dataloader in model.test_dataloader()]

    if trainer.use_ddp or trainer.use_ddp2:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()

    # test HPC loading / saving
    trainer.hpc_save(save_dir, logger)
    trainer.hpc_load(save_dir, on_gpu=on_gpu)

    clear_save_dir()


def get_hparams(continue_training=False, hpc_exp_number=0):
    root_dir = os.path.dirname(os.path.realpath(__file__))

    args = {
        'drop_prob': 0.2,
        'batch_size': 32,
        'in_features': 28 * 28,
        'learning_rate': 0.001 * 8,
        'optimizer_name': 'adam',
        'data_root': os.path.join(root_dir, 'mnist'),
        'out_features': 10,
        'hidden_dim': 1000}

    if continue_training:
        args['test_tube_do_checkpoint_load'] = True
        args['hpc_exp_number'] = hpc_exp_number

    hparams = Namespace(**args)
    return hparams


def get_model(use_test_model=False, lbfgs=False):
    # set up model with these hyperparams
    hparams = get_hparams()
    if lbfgs:
        setattr(hparams, 'optimizer_name', 'lbfgs')

    if use_test_model:
        model = LightningTestModel(hparams)
    else:
        model = LightningTemplateModel(hparams)

    return model, hparams


def get_test_tube_logger(debug=True, version=None):
    # set up logger object without actually saving logs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(root_dir, 'save_dir')
    logger = TestTubeLogger(save_dir, name='lightning_logs', debug=False, version=version)
    return logger


def init_save_dir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(root_dir, 'save_dir')

    if os.path.exists(save_dir):
        n = np.random.randint(0, 10000000, 1)[0]
        shutil.move(save_dir, save_dir + f'_{n}')

    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def clear_save_dir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(root_dir, 'save_dir')
    if os.path.exists(save_dir):
        n = np.random.randint(0, 10000000, 1)[0]
        shutil.move(save_dir, save_dir + f'_{n}')


def load_model(exp, save_dir, module_class=LightningTemplateModel):

    # load trained model
    tags_path = exp.get_data_path(exp.name, exp.version)
    checkpoint_folder = os.path.join(tags_path, 'checkpoints')
    tags_path = os.path.join(tags_path, 'meta_tags.csv')

    checkpoints = [x for x in os.listdir(checkpoint_folder) if '.ckpt' in x]
    weights_dir = os.path.join(checkpoint_folder, checkpoints[0])

    trained_model = module_class.load_from_metrics(weights_path=weights_dir,
                                                   tags_csv=tags_path)

    assert trained_model is not None, 'loading model failed'

    return trained_model


def run_prediction(dataloader, trained_model, dp=False):
    # run prediction on 1 batch
    for batch in dataloader:
        break

    x, y = batch
    x = x.view(x.size(0), -1)

    if dp:
        output = trained_model(batch, 0)
        acc = output['val_acc']
        acc = torch.mean(acc).item()

    else:
        y_hat = trained_model(x)

        # acc
        labels_hat = torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
        acc = torch.tensor(acc)
        acc = acc.item()

    assert acc > 0.50, f'this model is expected to get > 0.50 in test set (it got {acc})'


def assert_ok_val_acc(trainer):
    # this model should get 0.80+ acc
    acc = trainer.training_tqdm_dict['val_acc']
    assert acc > 0.50, f'model failed to get expected 0.50 validation accuracy. Got: {acc}'


def assert_ok_test_acc(trainer):
    # this model should get 0.80+ acc
    acc = trainer.training_tqdm_dict['test_acc']
    assert acc > 0.50, f'model failed to get expected 0.50 validation accuracy. Got: {acc}'


def can_run_gpu_test():
    if not torch.cuda.is_available():
        warnings.warn('test_multi_gpu_model_ddp cannot run.'
                      ' Rerun on a GPU node to run this test')
        return False
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_multi_gpu_model_ddp cannot run.'
                      ' Rerun on a node with 2+ GPUs to run this test')
        return False
    return True


def reset_seed():
    SEED = RANDOM_SEEDS.pop()
    torch.manual_seed(SEED)
    np.random.seed(SEED)


def set_random_master_port():
    port = RANDOM_PORTS.pop()
    os.environ['MASTER_PORT'] = str(port)


def init_checkpoint_callback(logger):
    exp = logger.experiment
    exp_path = exp.get_data_path(exp.name, exp.version)
    ckpt_dir = os.path.join(exp_path, 'checkpoints')
    checkpoint = ModelCheckpoint(ckpt_dir)
    return checkpoint


if __name__ == '__main__':
    pytest.main([__file__])
