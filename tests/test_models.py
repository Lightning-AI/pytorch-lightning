import os
import shutil
import warnings
from argparse import Namespace

import pytest
import numpy as np
import torch
from test_tube import Experiment, SlurmCluster

# sys.path += [os.path.abspath('..'), os.path.abspath('../..')]
from pytorch_lightning import Trainer
from pytorch_lightning.testing import LightningTestModel, NoValEndTestModel, NoValModel
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.utilities.debugging import MisconfigurationException
from pytorch_lightning.root_module import memory
from pytorch_lightning.models.trainer import reduce_distributed_output
from pytorch_lightning.root_module import model_saving
from examples import LightningTemplateModel

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------------
def test_multi_gpu_model_ddp():
    """
    Make sure DDP works
    :return:
    """
    if not torch.cuda.is_available():
        warnings.warn('test_multi_gpu_model_ddp cannot run.'
                      ' Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_multi_gpu_model_ddp cannot run.'
                      ' Rerun on a node with 2+ GPUs to run this test')
        return

    os.environ['MASTER_PORT'] = str(np.random.randint(12000, 19000, 1)[0])
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
    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      'Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      'Rerun on a node with 2+ GPUs to run this test')
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

    stopping = EarlyStopping(monitor='val_loss')
    trainer_options = dict(
        early_stop_callback=stopping,
        gradient_clip=1.0,
        overfit_pct=0.20,
        track_grad_norm=2,
        print_nan_grads=True,
        show_progress_bar=False,
        experiment=get_exp(),
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
    hparams = get_hparams()
    model = NoValModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
    exp = get_exp(False)
    exp.argparse(hparams)
    exp.save()

    trainer_options = dict(
        max_nb_epochs=1,
        cluster=SlurmCluster(),
        experiment=exp,
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
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_metrics(weights_path=new_weights_path,
                                                   tags_csv=tags_path, on_gpu=False)
    model_2.eval()

    # make prediction
    clear_save_dir()


def test_no_val_end_module():
    """
    Tests use case where trainer saves the model, and user loads it from tags independently
    :return:
    """
    hparams = get_hparams()
    model = NoValEndTestModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
    exp = get_exp(False)
    exp.argparse(hparams)
    exp.save()

    trainer_options = dict(
        max_nb_epochs=1,
        cluster=SlurmCluster(),
        experiment=exp,
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
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_metrics(weights_path=new_weights_path,
                                                   tags_csv=tags_path, on_gpu=False)
    model_2.eval()

    # make prediction
    clear_save_dir()


def test_simple_cpu():
    """
    Verify continue training session on CPU
    :return:
    """
    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
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
        gpus=[0],
        distributed_backend='dp',
        use_amp=True
    )

    run_gpu_model_test(trainer_options, model, hparams)


def test_cpu_restore_training():
    """
    Verify continue training session on CPU
    :return:
    """
    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
    test_exp_version = 10
    exp = get_exp(False, version=test_exp_version)
    exp.argparse(hparams)
    exp.save()

    trainer_options = dict(
        max_nb_epochs=2,
        val_check_interval=0.50,
        val_percent_check=0.2,
        train_percent_check=0.2,
        experiment=exp,
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
    new_exp = get_exp(False, version=test_exp_version)
    trainer_options = dict(
        max_nb_epochs=2,
        val_check_interval=0.50,
        val_percent_check=0.2,
        train_percent_check=0.2,
        experiment=new_exp,
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
        _ = [run_prediction(dataloader, trainer.model) for dataloader in trainer.val_dataloader]

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
    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      'Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      'Rerun on a node with 2+ GPUs to run this test')
        return

    os.environ['MASTER_PORT'] = str(np.random.randint(12000, 19000, 1)[0])

    hparams = get_hparams()
    model = LightningTestModel(hparams)

    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=1,
        gpus=[0, 1],
        distributed_backend='ddp',
        use_amp=True
    )

    run_gpu_model_test(trainer_options, model, hparams)


def test_cpu_slurm_save_load():
    """
    Verify model save/load/checkpoint on CPU
    :return:
    """
    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
    exp = get_exp(False)
    exp.argparse(hparams)
    exp.save()

    cluster_a = SlurmCluster()
    trainer_options = dict(
        max_nb_epochs=1,
        cluster=cluster_a,
        experiment=exp,
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
    for batch in model.test_dataloader:
        break

    x, y = batch
    x = x.view(x.size(0), -1)

    model.eval()
    pred_before_saving = model(x)

    # test registering a save function
    trainer.enable_auto_hpc_walltime_manager()

    # test HPC saving
    # simulate snapshot on slurm
    saved_filepath = trainer.hpc_save(save_dir, exp)
    assert os.path.exists(saved_filepath)

    # wipe-out trainer and model
    # retrain with not much data... this simulates picking training back up after slurm
    # we want to see if the weights come back correctly
    continue_tng_hparams = get_hparams(continue_training=True,
                                       hpc_exp_number=cluster_a.hpc_exp_number)
    trainer_options = dict(
        max_nb_epochs=1,
        cluster=SlurmCluster(continue_tng_hparams),
        experiment=exp,
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
    hparams = get_hparams()

    # save tags
    exp = get_exp(False)
    exp.tag({'some_str': 'a_str', 'an_int': 1, 'a_float': 2.0})
    exp.argparse(hparams)
    exp.save()

    # load tags
    tags_path = exp.get_data_path(exp.name, exp.version) + '/meta_tags.csv'
    tags = model_saving.load_hparams_from_tags_csv(tags_path)

    assert tags.batch_size == 32 and tags.hidden_dim == 1000

    clear_save_dir()


def test_dp_output_reduce():

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
    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
    exp = get_exp(False)
    exp.argparse(hparams)
    exp.save()

    trainer_options = dict(
        max_nb_epochs=1,
        cluster=SlurmCluster(),
        experiment=exp,
        checkpoint_callback=ModelCheckpoint(save_dir)
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # traning complete
    assert result == 1, 'amp + ddp model failed to complete'

    # make a prediction
    for batch in model.test_dataloader:
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
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')
    model_2 = LightningTestModel.load_from_metrics(weights_path=new_weights_path,
                                                   tags_csv=tags_path, on_gpu=False)
    model_2.eval()

    # make prediction
    # assert that both predictions are the same
    new_pred = model_2(x)
    assert torch.all(torch.eq(pred_before_saving, new_pred)).item() == 1

    clear_save_dir()


def test_model_freeze_unfreeze():
    hparams = get_hparams()
    model = LightningTestModel(hparams)

    model.freeze()
    model.unfreeze()


def test_amp_gpu_ddp_slurm_managed():
    """
    Make sure DDP + AMP work
    :return:
    """
    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      ' Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_ddp cannot run.'
                      ' Rerun on a node with 2+ GPUs to run this test')
        return

    # simulate setting slurm flags
    os.environ['MASTER_PORT'] = str(np.random.randint(12000, 19000, 1)[0])
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
    exp = get_exp(False)
    exp.argparse(hparams)
    exp.save()

    # exp file to get weights
    checkpoint = ModelCheckpoint(save_dir)

    # add these to the trainer options
    trainer_options['checkpoint_callback'] = checkpoint
    trainer_options['experiment'] = exp

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
    map_location = 'cuda:1'
    pretrained_model = load_model(exp, save_dir, True, map_location)

    # test model preds
    run_prediction(model.test_dataloader, pretrained_model)

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()

    # test HPC loading / saving
    trainer.hpc_save(save_dir, exp)
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

    trainer_options = dict(
        show_progress_bar=False,
        experiment=get_exp(),
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

    trainer_options = dict(
        show_progress_bar=False,
        experiment=get_exp(),
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

    trainer_options = dict(
        gradient_clip=1.0,
        overfit_pct=0.20,
        track_grad_norm=2,
        print_nan_grads=True,
        show_progress_bar=False,
        experiment=get_exp(),
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
        gpus=[0]
    )

    run_gpu_model_test(trainer_options, model, hparams)


def test_multi_gpu_model_dp():
    """
    Make sure DP works
    :return:
    """
    if not torch.cuda.is_available():
        warnings.warn('test_multi_gpu_model_dp cannot run.'
                      ' Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_multi_gpu_model_dp cannot run.'
                      ' Rerun on a node with 2+ GPUs to run this test')
        return
    model, hparams = get_model()
    trainer_options = dict(
        show_progress_bar=False,
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
    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_dp cannot run.'
                      ' Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_dp cannot run.'
                      ' Rerun on a node with 2+ GPUs to run this test')
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
    if not torch.cuda.is_available():
        warnings.warn('test_amp_gpu_ddp cannot run. Rerun on a GPU node to run this test')
        return
    if not torch.cuda.device_count() > 1:
        warnings.warn('test_amp_gpu_ddp cannot run. Rerun on a node with 2+ GPUs to run this test')
        return

    os.environ['MASTER_PORT'] = str(np.random.randint(12000, 19000, 1)[0])

    hparams = get_hparams()
    model = LightningTestModel(hparams, force_remove_distributed_sampler=True)

    exp = get_exp(True)
    exp.save()

    trainer = Trainer(
        experiment=exp,
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
    hparams = get_hparams()
    model = LightningTestModel(hparams)

    save_dir = init_save_dir()

    # exp file to get meta
    trainer_options = dict(
        max_nb_epochs=1,
        val_percent_check=0.1,
        train_percent_check=0.1,
    )

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # verify tng completed
    assert result == 1

    # verify there are 2 val loaders
    assert len(trainer.val_dataloader) == 2, 'Multiple val_dataloaders not initiated properly'

    # make sure predictions are good for each val set
    [run_prediction(dataloader, trainer.model) for dataloader in trainer.val_dataloader]


# ------------------------------------------------------------------------
# UTILS
# ------------------------------------------------------------------------
def run_gpu_model_test(trainer_options, model, hparams, on_gpu=True):
    save_dir = init_save_dir()

    # exp file to get meta
    exp = get_exp(False)
    exp.argparse(hparams)
    exp.save()

    # exp file to get weights
    checkpoint = ModelCheckpoint(save_dir)

    # add these to the trainer options
    trainer_options['checkpoint_callback'] = checkpoint
    trainer_options['experiment'] = exp

    # fit model
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)

    # correct result and ok accuracy
    assert result == 1, 'amp + ddp model failed to complete'

    # test model loading
    pretrained_model = load_model(exp, save_dir, on_gpu)

    # test model preds
    run_prediction(model.test_dataloader, pretrained_model)

    if trainer.use_ddp:
        # on hpc this would work fine... but need to hack it for the purpose of the test
        trainer.model = pretrained_model
        trainer.optimizers, trainer.lr_schedulers = pretrained_model.configure_optimizers()

    # test HPC loading / saving
    trainer.hpc_save(save_dir, exp)
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


def get_model(use_test_model=False):
    # set up model with these hyperparams
    hparams = get_hparams()

    if use_test_model:
        model = LightningTestModel(hparams)
    else:
        model = LightningTemplateModel(hparams)

    return model, hparams


def get_exp(debug=True, version=None):
    # set up exp object without actually saving logs
    root_dir = os.path.dirname(os.path.realpath(__file__))
    exp = Experiment(debug=debug, save_dir=root_dir, name='tests_tt_dir', version=version)
    return exp


def init_save_dir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(root_dir, 'save_dir')

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(save_dir, exist_ok=True)

    return save_dir


def clear_save_dir():
    root_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(root_dir, 'save_dir')
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)


def load_model(exp, save_dir, on_gpu, map_location=None):

    # load trained model
    tags_path = exp.get_data_path(exp.name, exp.version)
    tags_path = os.path.join(tags_path, 'meta_tags.csv')

    checkpoints = [x for x in os.listdir(save_dir) if '.ckpt' in x]
    weights_dir = os.path.join(save_dir, checkpoints[0])

    trained_model = LightningTemplateModel.load_from_metrics(weights_path=weights_dir,
                                                             tags_csv=tags_path,
                                                             on_gpu=on_gpu,
                                                             map_location=map_location)

    assert trained_model is not None, 'loading model failed'

    return trained_model


def run_prediction(dataloader, trained_model):
    # run prediction on 1 batch
    for batch in dataloader:
        break

    x, y = batch
    x = x.view(x.size(0), -1)

    y_hat = trained_model(x)

    # acc
    labels_hat = torch.argmax(y_hat, dim=1)
    val_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)
    val_acc = torch.tensor(val_acc)
    val_acc = val_acc.item()

    print(val_acc)

    assert val_acc > 0.50, 'this model is expected to get > 0.50 in test set (it got %f)' % val_acc


def assert_ok_acc(trainer):
    # this model should get 0.80+ acc
    acc = trainer.tng_tqdm_dic['val_acc']
    assert acc > 0.50, 'model failed to get expected 0.50 validation accuracy. Got: %f' % acc


if __name__ == '__main__':
    pytest.main([__file__])
