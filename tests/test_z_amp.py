import os
import warnings

import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.testing import (
    LightningTestModel,
)
from pytorch_lightning.utilities.debugging import MisconfigurationException
from .utils import (
    reset_seed, can_run_gpu_test, get_hparams, run_model_test, set_random_master_port, init_save_dir,
    get_test_tube_logger, init_checkpoint_callback, load_model, run_prediction, clear_save_dir, get_model
)


def test_amp_single_gpu():
    """
    Make sure DDP + AMP work
    :return:
    """
    reset_seed()

    if not can_run_gpu_test():
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

    run_model_test(trainer_options, model, hparams)


def test_no_amp_single_gpu():
    """
    Make sure DDP + AMP work
    :return:
    """
    reset_seed()

    if not can_run_gpu_test():
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
        run_model_test(trainer_options, model, hparams)


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

    run_model_test(trainer_options, model, hparams)


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
    pretrained_model = load_model(logger.experiment,
                                                trainer.checkpoint_callback.filepath)

    # test model preds
    for dataloader in trainer.get_test_dataloaders():
        run_prediction(dataloader, pretrained_model)

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
        run_model_test(trainer_options, model, hparams, on_gpu=False)


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
        run_model_test(trainer_options, model, hparams)


if __name__ == '__main__':
    pytest.main([__file__])
