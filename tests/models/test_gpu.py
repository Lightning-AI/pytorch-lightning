import os
import subprocess
import sys
from collections import namedtuple
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torchtext.data import Batch, Dataset, Example, Field, LabelField

import pytorch_lightning
import tests.base.develop_pipelines as tpipes
import tests.base.develop_utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.core import memory
from pytorch_lightning.trainer.distrib_parts import _parse_gpu_ids, determine_root_gpu_device
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.base import EvalModelTemplate
from tests.models.data.ddp import train_test_variations

PRETEND_N_OF_GPUS = 16


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_early_stop_dp(tmpdir):
    """Make sure DDP works. with early stopping"""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        early_stop_callback=True,
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='dp',
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_none_backend(tmpdir):
    """Make sure when using multiple GPUs the user can't use `distributed_backend = None`."""
    tutils.set_random_master_port()
    trainer_options = dict(
        default_root_dir=tmpdir,
        distributed_backend=None,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        gpus=2
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_early_stop_ddp_spawn(tmpdir):
    """Make sure DDP works. with early stopping"""
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        early_stop_callback=True,
        max_epochs=50,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='ddp_spawn',
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_model_dp(tmpdir):
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='dp',
        progress_bar_refresh_rate=0
    )

    model = EvalModelTemplate()

    tpipes.run_model_test(trainer_options, model)

    # test memory helper functions
    memory.get_memory_profile('min_max')


@pytest.mark.parametrize('cli_args', [
    pytest.param('--max_epochs 1 --gpus 2 --distributed_backend ddp'),
])
@pytest.mark.parametrize('variation', train_test_variations.get_variations())
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_model_ddp(tmpdir, cli_args, variation):
    """ Runs a basic training and test run with distributed_backend=ddp. """
    file = Path(train_test_variations.__file__).absolute()
    cli_args = cli_args.split(' ') if cli_args else []
    cli_args += ['--default_root_dir', str(tmpdir)]
    cli_args += ['--variation', variation]
    command = [sys.executable, str(file)] + cli_args

    # need to set the PYTHONPATH in case pytorch_lightning was not installed into the environment
    env = os.environ.copy()
    env['PYTHONPATH'] = f'{pytorch_lightning.__file__}:' + env.get('PYTHONPATH', '')

    # for running in ddp mode, we need to lauch it's own process or pytest will get stuck
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)

    std, err = p.communicate(timeout=60)
    std = std.decode('utf-8').strip()
    err = err.decode('utf-8').strip()
    assert std, f"{variation} produced no output"
    if p.returncode > 0:
        pytest.fail(err)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_multi_gpu_model_ddp_spawn(tmpdir):
    tutils.set_random_master_port()

    trainer_options = dict(
        default_root_dir=tmpdir,
        max_epochs=1,
        limit_train_batches=10,
        limit_val_batches=10,
        gpus=[0, 1],
        distributed_backend='ddp_spawn',
        progress_bar_refresh_rate=0
    )

    model = EvalModelTemplate()

    tpipes.run_model_test(trainer_options, model)

    # test memory helper functions
    memory.get_memory_profile('min_max')


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.parametrize('gpus', [1, [0], [1]])
def test_single_gpu_model(tmpdir, gpus):
    """Make sure single GPU works (DP mode)."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        gpus=gpus
    )

    model = EvalModelTemplate()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_ddp_all_dataloaders_passed_to_fit(tmpdir):
    """Make sure DDP works with dataloaders passed to fit()"""
    tutils.set_random_master_port()

    model = EvalModelTemplate()
    fit_options = dict(train_dataloader=model.train_dataloader(),
                       val_dataloaders=model.val_dataloader())

    trainer = Trainer(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        gpus=[0, 1],
        distributed_backend='ddp_spawn'
    )
    result = trainer.fit(model, **fit_options)
    assert result == 1, "DDP doesn't work with dataloaders passed to fit()."


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
        Trainer(gpus=gpus, distributed_backend=distributed_backend)


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
    assert _parse_gpu_ids(gpus) == expected_gpu_ids


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
        _parse_gpu_ids(gpus)


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize("gpus", [[1, 2, 19], -1, '-1'])
def test_parse_gpu_fail_on_non_existent_id(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        _parse_gpu_ids(gpus)


@pytest.mark.gpus_param_tests
def test_parse_gpu_fail_on_non_existent_id_2(mocked_device_count):
    with pytest.raises(MisconfigurationException):
        _parse_gpu_ids([1, 2, 19])


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize("gpus", [-1, '-1'])
def test_parse_gpu_returns_none_when_no_devices_are_available(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        _parse_gpu_ids(gpus)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_single_gpu_batch_parse():
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

    # namedtuple of tensor
    BatchType = namedtuple('BatchType', ['a', 'b'])
    batch = [BatchType(a=torch.rand(2, 3), b=torch.rand(2, 3)) for _ in range(2)]
    batch = trainer.transfer_batch_to_gpu(batch, 0)
    assert batch[0].a.device.index == 0
    assert batch[0].a.type() == 'torch.cuda.FloatTensor'

    # non-Tensor that has `.to()` defined
    class CustomBatchType:
        def __init__(self):
            self.a = torch.rand(2, 2)

        def to(self, *args, **kwargs):
            self.a = self.a.to(*args, **kwargs)
            return self

    batch = trainer.transfer_batch_to_gpu(CustomBatchType())
    assert batch.a.type() == 'torch.cuda.FloatTensor'

    # torchtext.data.Batch
    samples = [
        {'text': 'PyTorch Lightning is awesome!', 'label': 0},
        {'text': 'Please make it work with torchtext', 'label': 1}
    ]

    text_field = Field()
    label_field = LabelField()
    fields = {
        'text': ('text', text_field),
        'label': ('label', label_field)
    }

    examples = [Example.fromdict(sample, fields) for sample in samples]
    dataset = Dataset(
        examples=examples,
        fields=fields.values()
    )

    # Batch runs field.process() that numericalizes tokens, but it requires to build dictionary first
    text_field.build_vocab(dataset)
    label_field.build_vocab(dataset)

    batch = Batch(data=examples, dataset=dataset)
    batch = trainer.transfer_batch_to_gpu(batch, 0)

    assert batch.text.type() == 'torch.cuda.LongTensor'
    assert batch.label.type() == 'torch.cuda.LongTensor'


@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
def test_non_blocking():
    """ Tests that non_blocking=True only gets passed on torch.Tensor.to, but not on other objects. """
    trainer = Trainer()

    batch = torch.zeros(2, 3)
    with patch.object(batch, 'to', wraps=batch.to) as mocked:
        trainer.transfer_batch_to_gpu(batch, 0)
        mocked.assert_called_with(torch.device('cuda', 0), non_blocking=True)

    class BatchObject(object):

        def to(self, *args, **kwargs):
            pass

    batch = BatchObject()
    with patch.object(batch, 'to', wraps=batch.to) as mocked:
        trainer.transfer_batch_to_gpu(batch, 0)
        mocked.assert_called_with(torch.device('cuda', 0))
