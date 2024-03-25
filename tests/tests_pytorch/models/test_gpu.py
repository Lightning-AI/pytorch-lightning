# Copyright The Lightning AI team.
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
from collections import namedtuple
from unittest import mock
from unittest.mock import patch

import pytest
import torch
from lightning.fabric.plugins.environments import TorchElasticEnvironment
from lightning.fabric.utilities.device_parser import _parse_gpu_ids
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.accelerators import CPUAccelerator, CUDAAccelerator
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.utilities.exceptions import MisconfigurationException

import tests_pytorch.helpers.pipelines as tpipes
from tests_pytorch.helpers.datamodules import ClassifDataModule
from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.helpers.simple_models import ClassificationModel

PRETEND_N_OF_GPUS = 16


@RunIf(min_cuda_gpus=2, sklearn=True)
def test_multi_gpu_none_backend(tmp_path):
    """Make sure when using multiple GPUs the user can't use `accelerator = None`."""
    seed_everything(42)
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.2,
        "limit_val_batches": 0.2,
        "accelerator": "gpu",
        "strategy": "ddp_spawn",
        "devices": 2,
    }

    dm = ClassifDataModule()
    model = ClassificationModel()
    tpipes.run_model_test(trainer_options, model, dm)


@RunIf(min_cuda_gpus=2)
@pytest.mark.parametrize("devices", [1, [0], [1]])
def test_single_gpu_model(tmp_path, devices):
    seed_everything(42)
    trainer_options = {
        "default_root_dir": tmp_path,
        "enable_progress_bar": False,
        "max_epochs": 1,
        "limit_train_batches": 0.1,
        "limit_val_batches": 0.1,
        "accelerator": "gpu",
        "devices": devices,
        "strategy": "ddp_spawn",
    }

    model = BoringModel()
    tpipes.run_model_test(trainer_options, model)


@pytest.mark.parametrize(
    "devices",
    [
        1,
        3,
        [1, 2],
        [0, 1],
        -1,
        "-1",
    ],
)
def test_root_gpu_property_0_raising(mps_count_0, cuda_count_0, devices):
    """Test that asking for a GPU when none are available will result in a MisconfigurationException."""
    with pytest.raises(MisconfigurationException, match="No supported gpu backend found!"):
        Trainer(accelerator="gpu", devices=devices, strategy="ddp")


@mock.patch.dict(
    os.environ,
    {
        "CUDA_VISIBLE_DEVICES": "0",
        "LOCAL_RANK": "1",
        "GROUP_RANK": "1",
        "RANK": "3",
        "WORLD_SIZE": "4",
        "LOCAL_WORLD_SIZE": "2",
        "TORCHELASTIC_RUN_ID": "1",
    },
)
@pytest.mark.parametrize("devices", [[0, 1, 2], 2, "0,", [0, 2]])
def test_torchelastic_gpu_parsing(cuda_count_1, devices):
    """Ensure when using torchelastic and nproc_per_node is set to the default of 1 per GPU device that we omit
    sanitizing the gpus as only one of the GPUs is visible."""
    trainer = Trainer(accelerator="cuda", devices=devices)
    assert isinstance(trainer._accelerator_connector.cluster_environment, TorchElasticEnvironment)
    # when using gpu
    if _parse_gpu_ids(devices, include_cuda=True) is not None:
        assert isinstance(trainer.accelerator, CUDAAccelerator)
        assert trainer.num_devices == len(devices) if isinstance(devices, list) else devices
        assert trainer.device_ids == _parse_gpu_ids(devices, include_cuda=True)
    # fall back to cpu
    else:
        assert isinstance(trainer.accelerator, CPUAccelerator)
        assert trainer.num_devices == 1
        assert trainer.device_ids == [0]


@RunIf(min_cuda_gpus=1)
def test_single_gpu_batch_parse():
    trainer = Trainer(accelerator="gpu", devices=1)

    # non-transferrable types
    primitive_objects = [None, {}, [], 1.0, "x", [None, 2], {"x": (1, 2), "y": None}]
    for batch in primitive_objects:
        data = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
        assert data == batch

    # batch is just a tensor
    batch = torch.rand(2, 3)
    batch = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
    assert batch.device.index == 0
    assert batch.type() == "torch.cuda.FloatTensor"

    # tensor list
    batch = [torch.rand(2, 3), torch.rand(2, 3)]
    batch = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0].device.index == 0
    assert batch[0].type() == "torch.cuda.FloatTensor"
    assert batch[1].device.index == 0
    assert batch[1].type() == "torch.cuda.FloatTensor"

    # tensor list of lists
    batch = [[torch.rand(2, 3), torch.rand(2, 3)]]
    batch = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0][0].device.index == 0
    assert batch[0][0].type() == "torch.cuda.FloatTensor"
    assert batch[0][1].device.index == 0
    assert batch[0][1].type() == "torch.cuda.FloatTensor"

    # tensor dict
    batch = [{"a": torch.rand(2, 3), "b": torch.rand(2, 3)}]
    batch = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0]["a"].device.index == 0
    assert batch[0]["a"].type() == "torch.cuda.FloatTensor"
    assert batch[0]["b"].device.index == 0
    assert batch[0]["b"].type() == "torch.cuda.FloatTensor"

    # tuple of tensor list and list of tensor dict
    batch = ([torch.rand(2, 3) for _ in range(2)], [{"a": torch.rand(2, 3), "b": torch.rand(2, 3)} for _ in range(2)])
    batch = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0][0].device.index == 0
    assert batch[0][0].type() == "torch.cuda.FloatTensor"

    assert batch[1][0]["a"].device.index == 0
    assert batch[1][0]["a"].type() == "torch.cuda.FloatTensor"

    assert batch[1][0]["b"].device.index == 0
    assert batch[1][0]["b"].type() == "torch.cuda.FloatTensor"

    # namedtuple of tensor
    BatchType = namedtuple("BatchType", ["a", "b"])
    batch = [BatchType(a=torch.rand(2, 3), b=torch.rand(2, 3)) for _ in range(2)]
    batch = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
    assert batch[0].a.device.index == 0
    assert batch[0].a.type() == "torch.cuda.FloatTensor"

    # non-Tensor that has `.to()` defined
    class CustomBatchType:
        def __init__(self):
            self.a = torch.rand(2, 2)

        def to(self, *args, **kwargs):
            self.a = self.a.to(*args, **kwargs)
            return self

    batch = trainer.strategy.batch_to_device(CustomBatchType(), torch.device("cuda:0"))
    assert batch.a.type() == "torch.cuda.FloatTensor"


@RunIf(min_cuda_gpus=1)
def test_non_blocking():
    """Tests that non_blocking=True only gets passed on Tensor.to, but not on other objects."""
    trainer = Trainer()

    batch = torch.zeros(2, 3)
    with patch.object(batch, "to", wraps=batch.to) as mocked:
        batch = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
        mocked.assert_called_with(torch.device("cuda", 0), non_blocking=True)

    class BatchObject:
        def to(self, *args, **kwargs):
            pass

    batch = BatchObject()
    with patch.object(batch, "to", wraps=batch.to) as mocked:
        batch = trainer.strategy.batch_to_device(batch, torch.device("cuda:0"))
        mocked.assert_called_with(torch.device("cuda", 0))


@RunIf(min_cuda_gpus=1)
@pytest.mark.parametrize(
    ("strategy", "precision", "expected_dtype"),
    [
        ("auto", "16-mixed", torch.float32),
        ("auto", "16-true", torch.float16),
        pytest.param("deepspeed", "bf16-true", torch.bfloat16, marks=RunIf(deepspeed=True, bf16_cuda=True)),
    ],
)
def test_input_tensors_cast_before_transfer_to_device(strategy, precision, expected_dtype):
    class CustomBoringModel(BoringModel):
        def transfer_batch_to_device(self, batch, *args, **kwargs):
            assert batch.dtype == expected_dtype
            return super().transfer_batch_to_device(batch, *args, **kwargs)

    model = CustomBoringModel()
    trainer = Trainer(strategy=strategy, devices=1, precision=precision, barebones=True, max_steps=2)
    trainer.fit(model)
