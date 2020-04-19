import os

import pytest
import torch

import tests.base.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.core import memory
from pytorch_lightning.trainer.distrib_parts import parse_gpu_ids, determine_root_gpu_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate

PRETEND_N_OF_GPUS = 16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize('gpus', [1, [0], [1]])
def test_single_gpu_model(tmpdir, gpus):
    """Make sure single GPU works (DP mode)."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus=gpus
    )

    model = EvalModelTemplate()
    tutils.run_model_test(trainer_options, model)


@pytest.mark.spawn
@pytest.mark.parametrize("backend", ['dp', 'ddp', 'ddp2'])
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_model(tmpdir, backend):
    """Make sure DDP works."""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        train_percent_check=0.4,
        val_percent_check=0.2,
        gpus=[0, 1],
        distributed_backend=backend,
    )

    model = EvalModelTemplate()
    # tutils.run_model_test(trainer_options, model)
    trainer = Trainer(**trainer_options)
    result = trainer.fit(model)
    assert result

    # test memory helper functions
    memory.get_memory_profile('min_max')


@pytest.mark.spawn
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_ddp_all_dataloaders_passed_to_fit(tmpdir):
    """Make sure DDP works with dataloaders passed to fit()"""
    tutils.set_random_master_port()

    trainer_options = dict(default_root_dir=tmpdir,
                           progress_bar_refresh_rate=0,
                           max_epochs=1,
                           train_percent_check=0.1,
                           val_percent_check=0.1,
                           gpus=[0, 1],
                           distributed_backend='ddp')

    model = EvalModelTemplate()
    fit_options = dict(train_dataloader=model.train_dataloader(),
                       val_dataloaders=model.val_dataloader())

    trainer = Trainer(**trainer_options)
    result = trainer.fit(model, **fit_options)
    assert result == 1, "DDP doesn't work with dataloaders passed to fit()."


@pytest.mark.spawn
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_none_backend(tmpdir):
    """Make sure when using multiple GPUs the user can't use `distributed_backend = None`."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        train_percent_check=0.1,
        val_percent_check=0.1,
        gpus='-1'
    )

    model = EvalModelTemplate()
    with pytest.warns(UserWarning):
        tutils.run_model_test(trainer_options, model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_auto_move_data(tmpdir):
    """Make sure auto moving data works"""

    tutils.reset_seed()
    tutils.set_random_master_port()

    model, hparams = tutils.get_default_model()
    model = model.cuda(0)
    model.prepare_data()
    loader = model.train_dataloader()
    correct_device = [p.device for p in model.parameters()]
    assert all([correct_device[0] == d for d in correct_device]), 'All parameters must be on same device'
    correct_device = correct_device[0]
    for x, y in loader:
        x = x.view(x.size(0), -1)
        assert model(x).device == torch.device('cuda:0'), "Automoving data to same device as model failed"
        assert model(x.cuda(0)).device == torch.device('cuda:0'), "Automoving data to same device as model failed"


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


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(["gpus", "expected_num_gpus", "distributed_backend"], [
    pytest.param(None, 0, None, id="None - expect 0 gpu to use."),
    pytest.param(0, 0, None, id="Oth gpu, expect 1 gpu to use."),
    pytest.param(1, 1, None, id="1st gpu, expect 1 gpu to use."),
    pytest.param(-1, PRETEND_N_OF_GPUS, "ddp", id="-1 - use all gpus"),
    pytest.param('-1', PRETEND_N_OF_GPUS, "ddp", id="'-1' - use all gpus"),
    pytest.param(3, 3, "ddp", id="3rd gpu - 1 gpu to use (backend:ddp)")
])
def test_trainer_gpu_parse(mocked_device_count, gpus, expected_num_gpus, distributed_backend):
    assert Trainer(gpus=gpus, distributed_backend=distributed_backend).num_gpus == expected_num_gpus


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(["gpus", "expected_num_gpus", "distributed_backend"], [
    pytest.param(None, 0, None, id="None - expect 0 gpu to use."),
    pytest.param(None, 0, "ddp", id="None - expect 0 gpu to use."),
])
def test_trainer_num_gpu_0(mocked_device_count_0, gpus, expected_num_gpus, distributed_backend):
    assert Trainer(gpus=gpus, distributed_backend=distributed_backend).num_gpus == expected_num_gpus


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_root_gpu', "distributed_backend"], [
    pytest.param(None, None, "ddp", id="None is None"),
    pytest.param(0, None, "ddp", id="O gpus, expect gpu root device to be None."),
    pytest.param(1, 0, "ddp", id="1 gpu, expect gpu root device to be 0."),
    pytest.param(-1, 0, "ddp", id="-1 - use all gpus, expect gpu root device to be 0."),
    pytest.param('-1', 0, "ddp", id="'-1' - use all gpus, expect gpu root device to be 0."),
    pytest.param(3, 0, "ddp", id="3 gpus, expect gpu root device to be 0.(backend:ddp)")
])
def test_root_gpu_property(mocked_device_count, gpus, expected_root_gpu, distributed_backend):
    assert Trainer(gpus=gpus, distributed_backend=distributed_backend).root_gpu == expected_root_gpu


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_root_gpu', "distributed_backend"], [
    pytest.param(None, None, None, id="None is None"),
    pytest.param(None, None, "ddp", id="None is None"),
    pytest.param(0, None, "ddp", id="None is None"),
])
def test_root_gpu_property_0_passing(mocked_device_count_0, gpus, expected_root_gpu, distributed_backend):
    assert Trainer(gpus=gpus, distributed_backend=distributed_backend).root_gpu == expected_root_gpu


# Asking for a gpu when non are available will result in a MisconfigurationException
@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_root_gpu', "distributed_backend"], [
    pytest.param(1, None, "ddp"),
    pytest.param(3, None, "ddp"),
    pytest.param(3, None, "ddp"),
    pytest.param([1, 2], None, "ddp"),
    pytest.param([0, 1], None, "ddp"),
    pytest.param(-1, None, "ddp"),
    pytest.param('-1', None, "ddp")
])
def test_root_gpu_property_0_raising(mocked_device_count_0, gpus, expected_root_gpu, distributed_backend):
    with pytest.raises(MisconfigurationException):
        Trainer(gpus=gpus, distributed_backend=distributed_backend).root_gpu


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_root_gpu'], [
    pytest.param(None, None, id="No gpus, expect gpu root device to be None"),
    pytest.param([0], 0, id="Oth gpu, expect gpu root device to be 0."),
    pytest.param([1], 1, id="1st gpu, expect gpu root device to be 1."),
    pytest.param([3], 3, id="3rd gpu, expect gpu root device to be 3."),
    pytest.param([1, 2], 1, id="[1, 2] gpus, expect gpu root device to be 1."),
])
def test_determine_root_gpu_device(gpus, expected_root_gpu):
    assert determine_root_gpu_device(gpus) == expected_root_gpu


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_gpu_ids'], [
    pytest.param(None, None),
    pytest.param(0, None),
    pytest.param(1, [0]),
    pytest.param(3, [0, 1, 2]),
    pytest.param(-1, list(range(PRETEND_N_OF_GPUS)), id="-1 - use all gpus"),
    pytest.param([0], [0]),
    pytest.param([1, 3], [1, 3]),
    pytest.param('0', [0]),
    pytest.param('3', [3]),
    pytest.param('1, 3', [1, 3]),
    pytest.param('2,', [2]),
    pytest.param('-1', list(range(PRETEND_N_OF_GPUS)), id="'-1' - use all gpus"),
])
def test_parse_gpu_ids(mocked_device_count, gpus, expected_gpu_ids):
    assert parse_gpu_ids(gpus) == expected_gpu_ids


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus'], [
    pytest.param(0.1),
    pytest.param(-2),
    pytest.param(False),
    pytest.param([]),
    pytest.param([-1]),
    pytest.param([None]),
    pytest.param(['0']),
    pytest.param((0, 1)),
])
def test_parse_gpu_fail_on_unsupported_inputs(mocked_device_count, gpus):
    with pytest.raises(MisconfigurationException):
        parse_gpu_ids(gpus)


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize("gpus", [[1, 2, 19], -1, '-1'])
def test_parse_gpu_fail_on_non_existent_id(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        parse_gpu_ids(gpus)


@pytest.mark.gpus_param_tests
def test_parse_gpu_fail_on_non_existent_id_2(mocked_device_count):
    with pytest.raises(MisconfigurationException):
        parse_gpu_ids([1, 2, 19])


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize("gpus", [-1, '-1'])
def test_parse_gpu_returns_None_when_no_devices_are_available(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        parse_gpu_ids(gpus)
