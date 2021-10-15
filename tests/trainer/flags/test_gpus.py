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
import os
from unittest import mock

import pytest
import torch

import tests.helpers.pipelines as tpipes
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import device_parser
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers import BoringModel
from tests.helpers.runif import RunIf

PRETEND_N_OF_GPUS = 16


@RunIf(min_gpus=2)
@pytest.mark.parametrize("gpus", [1, [0], [1]])
def test_single_gpu_model(tmpdir, gpus):
    """Make sure single GPU works (DP mode)."""
    trainer_options = dict(
        default_root_dir=tmpdir,
        enable_progress_bar=False,
        max_epochs=1,
        limit_train_batches=0.1,
        limit_val_batches=0.1,
        gpus=gpus,
    )

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)


@pytest.fixture
def mocked_device_count(monkeypatch):
    def device_count():
        return PRETEND_N_OF_GPUS

    def is_available():
        return True

    monkeypatch.setattr(torch.cuda, "is_available", is_available)
    monkeypatch.setattr(torch.cuda, "device_count", device_count)


@pytest.fixture
def mocked_device_count_0(monkeypatch):
    def device_count():
        return 0

    monkeypatch.setattr(torch.cuda, "device_count", device_count)


@pytest.mark.parametrize(
    ["gpus", "expected_num_gpus", "distributed_backend"],
    [
        pytest.param(None, 0, None, id="None - expect 0 gpu to use."),
        pytest.param(0, 0, None, id="Oth gpu, expect 1 gpu to use."),
        pytest.param(1, 1, None, id="1st gpu, expect 1 gpu to use."),
        pytest.param(-1, PRETEND_N_OF_GPUS, "ddp", id="-1 - use all gpus"),
        pytest.param("-1", PRETEND_N_OF_GPUS, "ddp", id="'-1' - use all gpus"),
        pytest.param(3, 3, "ddp", id="3rd gpu - 1 gpu to use (backend:ddp)"),
    ],
)
def test_trainer_gpu_parse(mocked_device_count, gpus, expected_num_gpus, distributed_backend):
    assert Trainer(gpus=gpus, accelerator=distributed_backend).num_gpus == expected_num_gpus


@pytest.mark.parametrize(
    ["gpus", "expected_num_gpus", "distributed_backend"],
    [
        pytest.param(None, 0, None, id="None - expect 0 gpu to use."),
        pytest.param(None, 0, "ddp", id="None - expect 0 gpu to use."),
    ],
)
def test_trainer_num_gpu_0(mocked_device_count_0, gpus, expected_num_gpus, distributed_backend):
    assert Trainer(gpus=gpus, accelerator=distributed_backend).num_gpus == expected_num_gpus


@pytest.mark.parametrize(
    ["gpus", "expected_root_gpu", "distributed_backend"],
    [
        pytest.param(None, None, "ddp", id="None is None"),
        pytest.param(0, None, "ddp", id="O gpus, expect gpu root device to be None."),
        pytest.param(1, 0, "ddp", id="1 gpu, expect gpu root device to be 0."),
        pytest.param(-1, 0, "ddp", id="-1 - use all gpus, expect gpu root device to be 0."),
        pytest.param("-1", 0, "ddp", id="'-1' - use all gpus, expect gpu root device to be 0."),
        pytest.param(3, 0, "ddp", id="3 gpus, expect gpu root device to be 0.(backend:ddp)"),
    ],
)
def test_root_gpu_property(mocked_device_count, gpus, expected_root_gpu, distributed_backend):
    assert Trainer(gpus=gpus, accelerator=distributed_backend).root_gpu == expected_root_gpu


@pytest.mark.parametrize(
    ["gpus", "expected_root_gpu", "distributed_backend"],
    [
        pytest.param(None, None, None, id="None is None"),
        pytest.param(None, None, "ddp", id="None is None"),
        pytest.param(0, None, "ddp", id="None is None"),
    ],
)
def test_root_gpu_property_0_passing(mocked_device_count_0, gpus, expected_root_gpu, distributed_backend):
    assert Trainer(gpus=gpus, accelerator=distributed_backend).root_gpu == expected_root_gpu


# Asking for a gpu when non are available will result in a MisconfigurationException
@pytest.mark.parametrize(
    ["gpus", "expected_root_gpu", "distributed_backend"],
    [
        (1, None, "ddp"),
        (3, None, "ddp"),
        (3, None, "ddp"),
        ([1, 2], None, "ddp"),
        ([0, 1], None, "ddp"),
        (-1, None, "ddp"),
        ("-1", None, "ddp"),
    ],
)
def test_root_gpu_property_0_raising(mocked_device_count_0, gpus, expected_root_gpu, distributed_backend):
    with pytest.raises(MisconfigurationException):
        Trainer(gpus=gpus, accelerator=distributed_backend)


@pytest.mark.parametrize(
    ["gpus", "expected_root_gpu"],
    [
        pytest.param(None, None, id="No gpus, expect gpu root device to be None"),
        pytest.param([0], 0, id="Oth gpu, expect gpu root device to be 0."),
        pytest.param([1], 1, id="1st gpu, expect gpu root device to be 1."),
        pytest.param([3], 3, id="3rd gpu, expect gpu root device to be 3."),
        pytest.param([1, 2], 1, id="[1, 2] gpus, expect gpu root device to be 1."),
    ],
)
def test_determine_root_gpu_device(gpus, expected_root_gpu):
    assert device_parser.determine_root_gpu_device(gpus) == expected_root_gpu


@pytest.mark.parametrize(
    ["gpus", "expected_gpu_ids"],
    [
        (None, None),
        (0, None),
        (1, [0]),
        (3, [0, 1, 2]),
        pytest.param(-1, list(range(PRETEND_N_OF_GPUS)), id="-1 - use all gpus"),
        ([0], [0]),
        ([1, 3], [1, 3]),
        ((1, 3), [1, 3]),
        ("0", None),
        ("3", [0, 1, 2]),
        ("1, 3", [1, 3]),
        ("2,", [2]),
        pytest.param("-1", list(range(PRETEND_N_OF_GPUS)), id="'-1' - use all gpus"),
    ],
)
def test_parse_gpu_ids(mocked_device_count, gpus, expected_gpu_ids):
    assert device_parser.parse_gpu_ids(gpus) == expected_gpu_ids


@pytest.mark.parametrize("gpus", [0.1, -2, False, [], [-1], [None], ["0"], [0, 0]])
def test_parse_gpu_fail_on_unsupported_inputs(mocked_device_count, gpus):
    with pytest.raises(MisconfigurationException):
        device_parser.parse_gpu_ids(gpus)


@pytest.mark.parametrize("gpus", [[1, 2, 19]])
def test_parse_gpu_fail_on_non_existent_id(mocked_device_count_0, gpus):
    with pytest.raises(MisconfigurationException):
        device_parser.parse_gpu_ids(gpus)


def test_parse_gpu_fail_on_non_existent_id_2(mocked_device_count):
    with pytest.raises(MisconfigurationException):
        device_parser.parse_gpu_ids([1, 2, 19])


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "1",
        "RANK": "3",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "2",
    },
)
@mock.patch("torch.cuda.device_count", return_value=1)
@pytest.mark.parametrize("gpus", [[0, 1, 2], 2, "0"])
def test_torchelastic_gpu_parsing(mocked_device_count, gpus):
    """Ensure when using torchelastic and nproc_per_node is set to the default of 1 per GPU device That we omit
    sanitizing the gpus as only one of the GPUs is visible."""
    trainer = Trainer(gpus=gpus)
    assert isinstance(trainer.accelerator_connector.cluster_environment, TorchElasticEnvironment)
    assert trainer.accelerator_connector.parallel_device_ids == device_parser.parse_gpu_ids(gpus)
    assert trainer.gpus == gpus


@pytest.mark.parametrize("gpus", [-1, "-1"])
def test_all_gpus(tmpdir, gpus):
    """Testing that the -1 is stable for GPU machines also if GPU is missing."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmpdir,
        max_epochs=1,
        gpus=gpus,
    )
    trainer.fit(model)
    assert trainer.accelerator_connector.use_gpu == torch.cuda.is_available()
    assert trainer.accelerator_connector.num_gpus == torch.cuda.device_count()
