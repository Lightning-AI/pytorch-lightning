# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import operator
from collections import namedtuple
from unittest.mock import patch

import pytest
import torch

import tests.helpers.pipelines as tpipes
import tests.helpers.utils as tutils
from pytorch_lightning import Trainer
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _compare_version
from tests.helpers import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.imports import Batch, Dataset, Example, Field, LabelField
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel

PL_VERSION_LT_1_5 = _compare_version("pytorch_lightning", operator.lt, "1.5")
PRETEND_N_OF_GPUS = 16


@RunIf(min_gpus=2)
def test_multi_gpu_none_backend(tmpdir):
    """Make sure when using multiple GPUs the user can't use `distributed_backend = None`."""
    tutils.set_random_master_port()
    trainer_options = dict(
        default_root_dir=tmpdir,
        progress_bar_refresh_rate=0,
        max_epochs=1,
        limit_train_batches=0.2,
        limit_val_batches=0.2,
        gpus=2,
    )

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_gpus=2)
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

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)


@pytest.fixture
def mocked_device_count(monkeypatch):

    def device_count():
        return PRETEND_N_OF_GPUS

    def is_available():
        return True

    monkeypatch.setattr(torch.cuda, 'is_available', is_available)
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
    assert Trainer(gpus=gpus, accelerator=distributed_backend).num_gpus == expected_num_gpus


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(["gpus", "expected_num_gpus", "distributed_backend"], [
    pytest.param(None, 0, None, id="None - expect 0 gpu to use."),
    pytest.param(None, 0, "ddp", id="None - expect 0 gpu to use."),
])
def test_trainer_num_gpu_0(mocked_device_count_0, gpus, expected_num_gpus, distributed_backend):
    assert Trainer(gpus=gpus, accelerator=distributed_backend).num_gpus == expected_num_gpus


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
    assert Trainer(gpus=gpus, accelerator=distributed_backend).root_gpu == expected_root_gpu


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_root_gpu', "distributed_backend"], [
    pytest.param(None, None, None, id="None is None"),
    pytest.param(None, None, "ddp", id="None is None"),
    pytest.param(0, None, "ddp", id="None is None"),
])
def test_root_gpu_property_0_passing(mocked_device_count_0, gpus, expected_root_gpu, distributed_backend):
    assert Trainer(gpus=gpus, accelerator=distributed_backend).root_gpu == expected_root_gpu


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
        Trainer(gpus=gpus, accelerator=distributed_backend)


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_root_gpu'], [
    pytest.param(None, None, id="No gpus, expect gpu root device to be None"),
    pytest.param([0], 0, id="Oth gpu, expect gpu root device to be 0."),
    pytest.param([1], 1, id="1st gpu, expect gpu root device to be 1."),
    pytest.param([3], 3, id="3rd gpu, expect gpu root device to be 3."),
    pytest.param([1, 2], 1, id="[1, 2] gpus, expect gpu root device to be 1."),
])
def test_determine_root_gpu_device(gpus, expected_root_gpu):
    assert device_parser.determine_root_gpu_device(gpus) == expected_root_gpu


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus', 'expected_gpu_ids'], [
    pytest.param(None, None),
    pytest.param(0, None),
    pytest.param(1, [0]),
    pytest.param(3, [0, 1, 2]),
    pytest.param(-1, list(range(PRETEND_N_OF_GPUS)), id="-1 - use all gpus"),
    pytest.param([0], [0]),
    pytest.param([1, 3], [1, 3]),
    pytest.param((1, 3), [1, 3]),
    pytest.param('0', None, marks=pytest.mark.skipif(PL_VERSION_LT_1_5, reason="available from v1.5")),
    pytest.param('3', [0, 1, 2], marks=pytest.mark.skipif(PL_VERSION_LT_1_5, reason="available from v1.5")),
    pytest.param('1, 3', [1, 3]),
    pytest.param('2,', [2]),
    pytest.param('-1', list(range(PRETEND_N_OF_GPUS)), id="'-1' - use all gpus"),
])
def test_parse_gpu_ids(mocked_device_count, gpus, expected_gpu_ids):
    assert device_parser.parse_gpu_ids(gpus) == expected_gpu_ids


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize(['gpus'], [
    pytest.param(0.1),
    pytest.param(-2),
    pytest.param(False),
    pytest.param([]),
    pytest.param([-1]),
    pytest.param([None]),
    pytest.param(['0']),
])
def test_parse_gpu_fail_on_unsupported_inputs(mocked_device_count, gpus):
    with pytest.raises(MisconfigurationException):
        device_parser.parse_gpu_ids(gpus)


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize("gpus", [[1, 2, 19], -1, '-1'])
def test_parse_gpu_fail_on_non_existent_id(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        device_parser.parse_gpu_ids(gpus)


@pytest.mark.gpus_param_tests
def test_parse_gpu_fail_on_non_existent_id_2(mocked_device_count):
    with pytest.raises(MisconfigurationException):
        device_parser.parse_gpu_ids([1, 2, 19])


@pytest.mark.gpus_param_tests
@pytest.mark.parametrize("gpus", [-1, '-1'])
def test_parse_gpu_returns_none_when_no_devices_are_available(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        device_parser.parse_gpu_ids(gpus)


@RunIf(min_gpus=1)
def test_single_gpu_batch_parse():
    trainer = Trainer(gpus=1)

    # non-transferrable types
    primitive_objects = [None, {}, [], 1.0, "x", [None, 2], {"x": (1, 2), "y": None}]
    for batch in primitive_objects:
        data = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
        assert data == batch

    # batch is just a tensor
    batch = torch.rand(2, 3)
    batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
    assert batch.device.index == 0 and batch.type() == 'torch.cuda.FloatTensor'

    # tensor list
    batch = [torch.rand(2, 3), torch.rand(2, 3)]
    batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
    assert batch[0].device.index == 0 and batch[0].type() == 'torch.cuda.FloatTensor'
    assert batch[1].device.index == 0 and batch[1].type() == 'torch.cuda.FloatTensor'

    # tensor list of lists
    batch = [[torch.rand(2, 3), torch.rand(2, 3)]]
    batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
    assert batch[0][0].device.index == 0 and batch[0][0].type() == 'torch.cuda.FloatTensor'
    assert batch[0][1].device.index == 0 and batch[0][1].type() == 'torch.cuda.FloatTensor'

    # tensor dict
    batch = [{'a': torch.rand(2, 3), 'b': torch.rand(2, 3)}]
    batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
    assert batch[0]['a'].device.index == 0 and batch[0]['a'].type() == 'torch.cuda.FloatTensor'
    assert batch[0]['b'].device.index == 0 and batch[0]['b'].type() == 'torch.cuda.FloatTensor'

    # tuple of tensor list and list of tensor dict
    batch = ([torch.rand(2, 3) for _ in range(2)], [{'a': torch.rand(2, 3), 'b': torch.rand(2, 3)} for _ in range(2)])
    batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
    assert batch[0][0].device.index == 0 and batch[0][0].type() == 'torch.cuda.FloatTensor'

    assert batch[1][0]['a'].device.index == 0
    assert batch[1][0]['a'].type() == 'torch.cuda.FloatTensor'

    assert batch[1][0]['b'].device.index == 0
    assert batch[1][0]['b'].type() == 'torch.cuda.FloatTensor'

    # namedtuple of tensor
    BatchType = namedtuple('BatchType', ['a', 'b'])
    batch = [BatchType(a=torch.rand(2, 3), b=torch.rand(2, 3)) for _ in range(2)]
    batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
    assert batch[0].a.device.index == 0
    assert batch[0].a.type() == 'torch.cuda.FloatTensor'

    # non-Tensor that has `.to()` defined
    class CustomBatchType:

        def __init__(self):
            self.a = torch.rand(2, 2)

        def to(self, *args, **kwargs):
            self.a = self.a.to(*args, **kwargs)
            return self

    batch = trainer.accelerator.batch_to_device(CustomBatchType(), torch.device('cuda:0'))
    assert batch.a.type() == 'torch.cuda.FloatTensor'

    # torchtext.data.Batch
    samples = [{
        'text': 'PyTorch Lightning is awesome!',
        'label': 0
    }, {
        'text': 'Please make it work with torchtext',
        'label': 1
    }]

    text_field = Field()
    label_field = LabelField()
    fields = {'text': ('text', text_field), 'label': ('label', label_field)}

    examples = [Example.fromdict(sample, fields) for sample in samples]
    dataset = Dataset(examples=examples, fields=fields.values())

    # Batch runs field.process() that numericalizes tokens, but it requires to build dictionary first
    text_field.build_vocab(dataset)
    label_field.build_vocab(dataset)

    batch = Batch(data=examples, dataset=dataset)
    batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))

    assert batch.text.type() == 'torch.cuda.LongTensor'
    assert batch.label.type() == 'torch.cuda.LongTensor'


@RunIf(min_gpus=1)
def test_non_blocking():
    """ Tests that non_blocking=True only gets passed on torch.Tensor.to, but not on other objects. """
    trainer = Trainer()

    batch = torch.zeros(2, 3)
    with patch.object(batch, 'to', wraps=batch.to) as mocked:
        batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
        mocked.assert_called_with(torch.device('cuda', 0), non_blocking=True)

    class BatchObject(object):

        def to(self, *args, **kwargs):
            pass

    batch = BatchObject()
    with patch.object(batch, 'to', wraps=batch.to) as mocked:
        batch = trainer.accelerator.batch_to_device(batch, torch.device('cuda:0'))
        mocked.assert_called_with(torch.device('cuda', 0))
