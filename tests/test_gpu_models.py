import os
import pytest
import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)
from pytorch_lightning.root_module import memory
from pytorch_lightning.testing import (
    LightningTestModel,
)
from pytorch_lightning.trainer.dp_mixin import (
    parse_gpu_ids,
    determine_root_gpu_device,
)
from pytorch_lightning.utilities.debugging import MisconfigurationException
from . import testing_utils

PRETEND_N_OF_GPUS = 16


def test_multi_gpu_model_ddp2():
    """
    Make sure DDP2 works
    :return:
    """
    if not testing_utils.can_run_gpu_test():
        return

    testing_utils.reset_seed()
    testing_utils.set_random_master_port()

    model, hparams = testing_utils.get_model()
    trainer_options = dict(
        show_progress_bar=True,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        gpus=2,
        weights_summary=None,
        distributed_backend='ddp2'
    )

    testing_utils.run_gpu_model_test(trainer_options, model, hparams)


def test_multi_gpu_model_ddp():
    """
    Make sure DDP works
    :return:
    """
    if not testing_utils.can_run_gpu_test():
        return

    testing_utils.reset_seed()
    testing_utils.set_random_master_port()

    model, hparams = testing_utils.get_model()
    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        gpus=[0, 1],
        distributed_backend='ddp'
    )

    testing_utils.run_gpu_model_test(trainer_options, model, hparams)


def test_optimizer_return_options():
    testing_utils.reset_seed()

    trainer = Trainer()
    model, hparams = testing_utils.get_model()

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


def test_cpu_slurm_save_load():
    """
    Verify model save/load/checkpoint on CPU
    :return:
    """
    testing_utils.reset_seed()

    hparams = testing_utils.get_hparams()
    model = LightningTestModel(hparams)

    save_dir = testing_utils.init_save_dir()

    # logger file to get meta
    logger = testing_utils.get_test_tube_logger(False)

    version = logger.version

    trainer_options = dict(
        max_nb_epochs=1,
        logger=logger,
        checkpoint_callback=ModelCheckpoint(save_dir, saving_mode='all')
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
    logger = testing_utils.get_test_tube_logger(False, version=version)

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

    testing_utils.clear_save_dir()


def test_multi_gpu_none_backend():
    """
    Make sure when using multiple GPUs the user can't use
    distributed_backend = None
    :return:
    """
    testing_utils.reset_seed()

    if not testing_utils.can_run_gpu_test():
        return

    model, hparams = testing_utils.get_model()
    trainer_options = dict(
        show_progress_bar=False,
        max_nb_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus='-1'
    )

    with pytest.raises(MisconfigurationException):
        testing_utils.run_gpu_model_test(trainer_options, model, hparams)


def test_multi_gpu_model_dp():
    """
    Make sure DP works
    :return:
    """
    testing_utils.reset_seed()

    if not testing_utils.can_run_gpu_test():
        return

    model, hparams = testing_utils.get_model()
    trainer_options = dict(
        show_progress_bar=False,
        distributed_backend='dp',
        max_nb_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus='-1'
    )

    testing_utils.run_gpu_model_test(trainer_options, model, hparams)

    # test memory helper functions
    memory.get_gpu_memory_map()


def test_ddp_sampler_error():
    """
    Make sure DDP + AMP work
    :return:
    """
    if not testing_utils.can_run_gpu_test():
        return

    testing_utils.reset_seed()
    testing_utils.set_random_master_port()

    hparams = testing_utils.get_hparams()
    model = LightningTestModel(hparams, force_remove_distributed_sampler=True)

    logger = testing_utils.get_test_tube_logger(True)

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

    testing_utils.clear_save_dir()


@pytest.fixture
def mocked_device_count(monkeypatch):
    def device_count():
        return PRETEND_N_OF_GPUS

    monkeypatch.setattr(torch.cuda, 'device_count', device_count)


@pytest.fixture
def mocked_device_count_0(monkeypatch):
    def device_count():
        return 0

    monkeypatch.setattr(torch.cuda, 'device_count', device_count)


test_num_gpus_data = [
    pytest.param(None, 0, None, id="None - expect 0 gpu to use."),
    pytest.param(0, 0, None, id="Oth gpu, expect 1 gpu to use."),
    pytest.param(1, 1, None, id="1st gpu, expect 1 gpu to use."),
    pytest.param(-1, PRETEND_N_OF_GPUS, "ddp", id="-1 - use all gpus"),
    pytest.param('-1', PRETEND_N_OF_GPUS, "ddp", id="'-1' - use all gpus"),
    pytest.param(3, 3, "ddp", id="3rd gpu - 1 gpu to use (backend:ddp)")
]


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(["gpus", "expected_num_gpus", "distributed_backend"], test_num_gpus_data)
def test_trainer_gpu_parse(mocked_device_count, gpus, expected_num_gpus, distributed_backend):
    assert Trainer(gpus=gpus, distributed_backend=distributed_backend).num_gpus == expected_num_gpus


test_num_gpus_data_0 = [
    pytest.param(None, 0, None, id="None - expect 0 gpu to use."),
    pytest.param(None, 0, "ddp", id="None - expect 0 gpu to use."),
]


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(["gpus", "expected_num_gpus", "distributed_backend"], test_num_gpus_data_0)
def test_trainer_num_gpu_0(mocked_device_count_0, gpus, expected_num_gpus, distributed_backend):
    assert Trainer(gpus=gpus, distributed_backend=distributed_backend).num_gpus == expected_num_gpus


test_root_gpu_data = [
    pytest.param(None, None, "ddp", id="None is None"),
    pytest.param(0, None, "ddp", id="O gpus, expect gpu root device to be None."),
    pytest.param(1, 0, "ddp", id="1 gpu, expect gpu root device to be 0."),
    pytest.param(-1, 0, "ddp", id="-1 - use all gpus, expect gpu root device to be 0."),
    pytest.param('-1', 0, "ddp", id="'-1' - use all gpus, expect gpu root device to be 0."),
    pytest.param(3, 0, "ddp", id="3 gpus, expect gpu root device to be 0.(backend:ddp)")]


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_root_gpu', "distributed_backend"], test_root_gpu_data)
def test_root_gpu_property(mocked_device_count, gpus, expected_root_gpu, distributed_backend):
    assert Trainer(gpus=gpus, distributed_backend=distributed_backend).root_gpu == expected_root_gpu


test_root_gpu_data_for_0_devices_passing = [
    pytest.param(None, None, None, id="None is None"),
    pytest.param(None, None, "ddp", id="None is None"),
    pytest.param(0, None, "ddp", id="None is None"),
]


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize([
    'gpus', 'expected_root_gpu', "distributed_backend"], test_root_gpu_data_for_0_devices_passing)
def test_root_gpu_property_0_passing(
        mocked_device_count_0, gpus, expected_root_gpu, distributed_backend):
    assert Trainer(gpus=gpus, distributed_backend=distributed_backend).root_gpu == expected_root_gpu


# Asking for a gpu when non are available will result in a MisconfigurationException
test_root_gpu_data_for_0_devices_raising = [
    pytest.param(1, None, "ddp"),
    pytest.param(3, None, "ddp"),
    pytest.param(3, None, "ddp"),
    pytest.param([1, 2], None, "ddp"),
    pytest.param([0, 1], None, "ddp"),
    pytest.param(-1, None, "ddp"),
    pytest.param('-1', None, "ddp")
]


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize([
    'gpus', 'expected_root_gpu', "distributed_backend"], test_root_gpu_data_for_0_devices_raising)
def test_root_gpu_property_0_raising(
        mocked_device_count_0, gpus, expected_root_gpu, distributed_backend):
    with pytest.raises(MisconfigurationException):
        Trainer(gpus=gpus, distributed_backend=distributed_backend).root_gpu


test_determine_root_gpu_device_data = [
    pytest.param(None, None, id="No gpus, expect gpu root device to be None"),
    pytest.param([0], 0, id="Oth gpu, expect gpu root device to be 0."),
    pytest.param([1], 1, id="1st gpu, expect gpu root device to be 1."),
    pytest.param([3], 3, id="3rd gpu, expect gpu root device to be 3."),
    pytest.param([1, 2], 1, id="[1, 2] gpus, expect gpu root device to be 1."),
]


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_root_gpu'], test_determine_root_gpu_device_data)
def test_determine_root_gpu_device(gpus, expected_root_gpu):
    assert determine_root_gpu_device(gpus) == expected_root_gpu


test_parse_gpu_ids_data = [
    pytest.param(None, None),
    pytest.param(0, None),
    pytest.param(1, [0]),
    pytest.param(-1, list(range(PRETEND_N_OF_GPUS)), id="-1 - use all gpus"),
    pytest.param('-1', list(range(PRETEND_N_OF_GPUS)), id="'-1' - use all gpus"),
    pytest.param(3, [0, 1, 2])]


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_gpu_ids'], test_parse_gpu_ids_data)
def test_parse_gpu_ids(mocked_device_count, gpus, expected_gpu_ids):
    assert parse_gpu_ids(gpus) == expected_gpu_ids


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize("gpus", [[1, 2, 19], -1, '-1'])
def test_parse_gpu_fail_on_non_existant_id(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        parse_gpu_ids(gpus)


@pytest.mark.gpus_param_tests
def test_parse_gpu_fail_on_non_existant_id_2(mocked_device_count):
    with pytest.raises(MisconfigurationException):
        parse_gpu_ids([1, 2, 19])


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize("gpus", [-1, '-1'])
def test_parse_gpu_returns_None_when_no_devices_are_available(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        parse_gpu_ids(gpus)


if __name__ == '__main__':
    pytest.main([__file__])
