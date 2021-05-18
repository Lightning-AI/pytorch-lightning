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

from pytorch_lightning import Trainer
from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin, SingleDevicePlugin, SingleTPUPlugin, TPUSpawnPlugin
from tests.helpers.runif import RunIf


def test_single_cpu():
    """Tests if on_gpu and on_tpu is set correctly for single cpu plugin."""
    trainer = Trainer()
    assert isinstance(trainer.training_type_plugin, SingleDevicePlugin)
    assert not trainer.training_type_plugin.on_gpu
    assert not trainer.training_type_plugin.on_tpu


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"})
@mock.patch("torch.cuda.device_count", return_value=1)
@mock.patch("torch.cuda.is_available", return_value=True)
def test_single_gpu(device_count_mock, mock_cuda_available):
    """Tests if on_gpu and on_tpu is set correctly for single gpu plugin."""
    trainer = Trainer(gpus=1)
    assert isinstance(trainer.training_type_plugin, SingleDevicePlugin)
    assert trainer.training_type_plugin.on_gpu
    assert not trainer.training_type_plugin.on_tpu


@mock.patch("torch.cuda.is_available", return_value=False)
def test_ddp_cpu(mock_cuda_available):
    """Tests if on_gpu and on_tpu is set correctly for ddp_cpu plugin."""
    trainer = Trainer(num_processes=2)
    assert isinstance(trainer.training_type_plugin, DDPSpawnPlugin)
    assert not trainer.training_type_plugin.on_gpu
    assert not trainer.training_type_plugin.on_tpu


@mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"})
@mock.patch("torch.cuda.device_count", return_value=2)
@mock.patch("torch.cuda.is_available", return_value=True)
def test_ddp_multi_gpu(device_count_mock, mock_cuda_available):
    """Tests if on_gpu and on_tpu is set correctly for multi gpu ddp plugin."""
    trainer = Trainer(
        gpus=2,
        accelerator="ddp",
    )
    assert isinstance(trainer.training_type_plugin, DDPPlugin)
    assert trainer.training_type_plugin.on_gpu
    assert not trainer.training_type_plugin.on_tpu


@RunIf(tpu=True)
def test_single_tpu():
    """Tests in_gpu and on_tpu is set correctly for single tpu plugin."""
    trainer = Trainer(tpu_cores=1)
    assert isinstance(trainer.training_type_plugin, SingleTPUPlugin)
    assert not trainer.training_type_plugin.on_gpu
    assert trainer.training_type_plugin.on_tpu


@RunIf(tpu=True)
def test_multi_tpu():
    """Tests in_gpu and on_tpu is set correctly for multi tpu plugin."""
    trainer = Trainer(tpu_cores=8)
    assert isinstance(trainer.training_type_plugin, TPUSpawnPlugin)
    assert not trainer.training_type_plugin.on_gpu
    assert trainer.training_type_plugin.on_tpu
