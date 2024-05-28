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

import logging
import os
from unittest import mock

import pytest
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomIterableDataset
from lightning.pytorch.strategies import SingleDeviceXLAStrategy
from torch.utils.data import DataLoader

from tests_pytorch.conftest import mock_cuda_count
from tests_pytorch.helpers.runif import RunIf


def test_num_stepping_batches_basic():
    """Test number of stepping batches in a general case."""
    max_epochs = 2
    trainer = Trainer(max_epochs=max_epochs)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == 64 * max_epochs


def test_num_stepping_batches_raises_info_with_no_dataloaders_loaded(caplog):
    """Test that an info message is generated when dataloaders are loaded explicitly if they are not already
    configured."""
    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)

    # artificially setup the data
    trainer.fit_loop.setup_data()

    with caplog.at_level(logging.INFO):
        assert trainer.estimated_stepping_batches == 64

    message = "to estimate number of stepping batches"
    assert message not in caplog.text

    trainer = Trainer(max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)

    with caplog.at_level(logging.INFO):
        assert trainer.estimated_stepping_batches == 64

    assert message in caplog.text


def test_num_stepping_batches_iterable_dataset():
    """Test the stepping batches with iterable dataset configured with max steps."""
    max_steps = 1000
    trainer = Trainer(max_steps=max_steps)
    model = BoringModel()
    train_dl = DataLoader(RandomIterableDataset(size=7, count=int(1e10)))
    trainer._data_connector.attach_data(model, train_dataloaders=train_dl)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == max_steps


def test_num_stepping_batches_infinite_training():
    """Test that stepping batches is "inf" when `Trainer` is configured for infinite training."""
    trainer = Trainer(max_steps=-1, max_epochs=-1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == float("inf")


@pytest.mark.parametrize("max_steps", [2, 100])
def test_num_stepping_batches_with_max_steps(max_steps, tmp_path):
    """Test stepping batches with `max_steps`."""
    trainer = Trainer(max_steps=max_steps, default_root_dir=tmp_path, logger=False, enable_checkpointing=False)
    model = BoringModel()
    trainer.fit(model)
    assert trainer.estimated_stepping_batches == max_steps


@pytest.mark.parametrize(("accumulate_grad_batches", "expected_steps"), [(2, 32), (3, 22)])
def test_num_stepping_batches_accumulate_gradients(accumulate_grad_batches, expected_steps):
    """Test the total stepping batches when accumulation grad batches is configured."""
    trainer = Trainer(max_epochs=1, accumulate_grad_batches=accumulate_grad_batches)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == expected_steps


@RunIf(mps=False)
@pytest.mark.parametrize(
    ("trainer_kwargs", "estimated_steps"),
    [
        ({"strategy": "ddp", "num_nodes": 1}, 10),
        ({"strategy": "ddp", "num_nodes": 2}, 5),
        ({"strategy": "ddp", "num_nodes": 3}, 4),
        ({"strategy": "ddp", "num_nodes": 4}, 3),
    ],
)
def test_num_stepping_batches_gpu(trainer_kwargs, estimated_steps, monkeypatch):
    """Test stepping batches with GPU strategies."""
    num_devices_per_node = 7
    mock_cuda_count(monkeypatch, num_devices_per_node)
    trainer = Trainer(max_epochs=1, devices=num_devices_per_node, accelerator="gpu", **trainer_kwargs)

    # set the `parallel_devices` to cpu to run the test on CPU and take `num_nodes`` into consideration
    # because we can't run on multi-node in ci
    trainer.strategy.parallel_devices = [torch.device("cpu", index=i) for i in range(num_devices_per_node)]

    model = BoringModel()
    trainer._data_connector.attach_data(model)
    trainer.strategy.connect(model)
    assert trainer.estimated_stepping_batches == estimated_steps


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_num_stepping_batches_with_tpu_single():
    """Test stepping batches with the single-core TPU strategy."""
    trainer = Trainer(accelerator="tpu", devices=1, max_epochs=1)
    model = BoringModel()
    trainer._data_connector.attach_data(model)
    assert isinstance(trainer.strategy, SingleDeviceXLAStrategy)
    trainer.strategy.connect(model)
    expected = len(model.train_dataloader())
    assert trainer.estimated_stepping_batches == expected


class MultiprocessModel(BoringModel):
    def on_train_start(self):
        assert self.trainer.estimated_stepping_batches == len(self.train_dataloader()) // self.trainer.world_size


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_num_stepping_batches_with_tpu_multi():
    """Test stepping batches with the TPU strategy across multiple devices."""
    trainer = Trainer(accelerator="tpu", devices="auto", max_epochs=1, logger=False, enable_checkpointing=False)
    model = MultiprocessModel()
    trainer.fit(model)
