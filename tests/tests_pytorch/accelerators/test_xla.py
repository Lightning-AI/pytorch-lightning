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
# limitations under the License
import collections
import os
from copy import deepcopy
from unittest import mock
from unittest.mock import MagicMock, call, patch

import lightning.fabric
import pytest
import torch
from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import CPUAccelerator, XLAAccelerator
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.plugins import Precision, XLACheckpointIO, XLAPrecision
from lightning.pytorch.strategies import DDPStrategy, XLAStrategy
from lightning.pytorch.utilities import find_shared_parameters
from torch import nn
from torch.utils.data import DataLoader

from tests_pytorch.helpers.runif import RunIf
from tests_pytorch.trainer.connectors.test_accelerator_connector import DeviceMock
from tests_pytorch.trainer.optimization.test_manual_optimization import assert_emtpy_grad


class WeightSharingModule(BoringModel):
    def __init__(self):
        super(BoringModel, self).__init__()
        self.layer_1 = nn.Linear(32, 10, bias=False)
        self.layer_2 = nn.Linear(10, 32, bias=False)
        self.layer_3 = nn.Linear(32, 10, bias=False)
        self.layer_3.weight = self.layer_1.weight

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


@RunIf(tpu=True, standalone=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_resume_training_on_cpu(tmp_path):
    """Checks if training can be resumed from a saved checkpoint on CPU."""
    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(max_epochs=1, accelerator="tpu", devices="auto", default_root_dir=tmp_path)
    trainer.fit(model)

    if trainer.world_size != trainer.num_devices:
        # we're in multinode. unless the filesystem is shared, only the main node will have access to the checkpoint
        # since we cannot know this, the code below needs to be skipped
        return

    model_path = trainer.checkpoint_callback.best_model_path

    # Verify saved Tensors are on CPU
    ckpt = torch.load(model_path)
    weight_tensor = list(ckpt["state_dict"].values())[0]
    assert weight_tensor.device == torch.device("cpu")

    # Verify that training is resumed on CPU
    trainer = Trainer(max_epochs=1, default_root_dir=tmp_path)
    trainer.fit(model, ckpt_path=model_path)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_if_test_works_after_train(tmp_path):
    """Ensure that .test() works after .fit()"""
    model = BoringModel()
    trainer = Trainer(max_epochs=1, accelerator="tpu", devices="auto", default_root_dir=tmp_path, fast_dev_run=True)
    trainer.fit(model)
    out = trainer.test(model)
    assert len(out) == 1


@RunIf(skip_windows=True)
def test_accelerator_cpu_when_tpu_available(tpu_available):
    assert XLAAccelerator.is_available()
    trainer = Trainer(accelerator="cpu", devices=8)
    assert isinstance(trainer.accelerator, CPUAccelerator)


@RunIf(skip_windows=True)
@pytest.mark.parametrize(("accelerator", "devices"), [("auto", 8), ("auto", "auto"), ("tpu", "auto")])
def test_accelerator_tpu(accelerator, devices, tpu_available):
    assert XLAAccelerator.is_available()

    trainer = Trainer(accelerator=accelerator, devices=devices)
    assert isinstance(trainer.accelerator, XLAAccelerator)
    assert isinstance(trainer.strategy, XLAStrategy)


class ManualOptimizationModel(BoringModel):
    count = 0
    called = collections.defaultdict(int)

    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    @property
    def should_update(self):
        return self.count % 2 == 0

    def on_train_batch_start(self, batch, batch_idx):
        self.called["on_train_batch_start"] += 1
        self.weight_before = self.layer.weight.clone()

    def training_step(self, batch, batch_idx):
        self.called["training_step"] += 1
        opt = self.optimizers()
        loss = self.step(batch)

        if self.should_update:
            self.manual_backward(loss)
            opt.step()
            opt.zero_grad()
        return loss

    def on_train_batch_end(self, *_):
        self.called["on_train_batch_end"] += 1
        after_before = self.layer.weight.clone()
        if self.should_update:
            assert not torch.equal(self.weight_before, after_before), self.count
        else:
            assert torch.equal(self.weight_before, after_before)
        assert_emtpy_grad(self.layer.weight.grad)
        self.count += 1

    def on_train_start(self):
        opt = self.optimizers()
        self.opt_step_patch = patch.object(opt, "step", wraps=opt.step)
        self.opt_step_mock = self.opt_step_patch.start()

    def on_train_end(self):
        # this might fail if run in an environment with too many ranks, as the total
        # length of the dataloader will be distrbuted among them and then each rank might not do 3 steps
        assert self.called["training_step"] == 3
        assert self.called["on_train_batch_start"] == 3
        assert self.called["on_train_batch_end"] == 3

        self.opt_step_patch.stop()
        assert self.opt_step_mock.call_count == 2


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_manual_optimization_tpus(tmp_path):
    model = ManualOptimizationModel()
    model_copy = deepcopy(model)

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmp_path,
        limit_train_batches=3,
        limit_test_batches=0,
        limit_val_batches=0,
        accelerator="tpu",
        devices="auto",
    )
    trainer.fit(model)

    for param, param_copy in zip(model.parameters(), model_copy.parameters()):
        assert not torch.equal(param.cpu().data, param_copy.data)


def test_strategy_choice_tpu_str_ddp_spawn(tpu_available):
    with pytest.raises(ValueError, match="XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy`"):
        Trainer(strategy="ddp_spawn", accelerator="tpu", devices=8)


@RunIf(skip_windows=True)
@mock.patch("lightning.pytorch.strategies.xla.XLAStrategy.set_world_ranks")
def test_strategy_choice_tpu_str_xla_debug(_, tpu_available):
    trainer = Trainer(strategy="xla_debug", accelerator="tpu", devices=8)
    assert isinstance(trainer.strategy, XLAStrategy)


@RunIf(tpu=True)
def test_strategy_choice_tpu_strategy():
    trainer = Trainer(strategy=XLAStrategy(), accelerator="tpu", devices="auto")
    assert isinstance(trainer.strategy, XLAStrategy)


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_auto_parameters_tying_tpus(tmp_path):
    model = WeightSharingModule()
    shared_params = find_shared_parameters(model)

    assert shared_params[0] == ["layer_1.weight", "layer_3.weight"]

    trainer = Trainer(default_root_dir=tmp_path, limit_train_batches=3, accelerator="tpu", devices="auto", max_epochs=1)
    trainer.fit(model)

    assert torch.equal(model.layer_1.weight, model.layer_3.weight)


class SubModule(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x)


class NestedModule(BoringModel):
    def __init__(self):
        super(BoringModel, self).__init__()
        self.layer = nn.Linear(32, 10, bias=False)
        self.net_a = SubModule(self.layer)
        self.layer_2 = nn.Linear(10, 32, bias=False)
        self.net_b = SubModule(self.layer)

    def forward(self, x):
        x = self.net_a(x)
        x = self.layer_2(x)
        x = self.net_b(x)
        return x


@RunIf(tpu=True)
@mock.patch.dict(os.environ, os.environ.copy(), clear=True)
def test_auto_parameters_tying_tpus_nested_module(tmp_path):
    model = NestedModule()
    trainer = Trainer(default_root_dir=tmp_path, limit_train_batches=3, accelerator="tpu", devices="auto", max_epochs=1)
    trainer.fit(model)

    assert torch.all(torch.eq(model.net_a.layer.weight, model.net_b.layer.weight))


def test_tpu_invalid_raises(tpu_available, mps_count_0):
    strategy = DDPStrategy(accelerator=XLAAccelerator(), precision_plugin=XLAPrecision())
    with pytest.raises(ValueError, match="XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy`"):
        Trainer(strategy=strategy, devices=8)

    accelerator = XLAAccelerator()
    with pytest.raises(TypeError, match="can only work with the `XLAPrecision` plugin"):
        XLAStrategy(accelerator=accelerator, precision_plugin=Precision())

    accelerator = XLAAccelerator()
    strategy = DDPStrategy(accelerator=accelerator, precision_plugin=XLAPrecision())
    with pytest.raises(
        ValueError, match="The `XLAAccelerator` can only be used with a `SingleDeviceXLAStrategy` or `XLAStrategy"
    ):
        Trainer(strategy=strategy, devices=8)


@RunIf(skip_windows=True)
@mock.patch("lightning.pytorch.strategies.xla.XLAStrategy.set_world_ranks")
def test_xla_checkpoint_plugin_being_default(_, tpu_available):
    trainer = Trainer(accelerator="tpu", devices=8)
    assert isinstance(trainer.strategy.checkpoint_io, XLACheckpointIO)


@RunIf(tpu=True)
@patch("lightning.pytorch.strategies.xla.XLAStrategy.root_device")
def test_xla_mp_device_dataloader_attribute(_, monkeypatch):
    dataset = RandomDataset(32, 64)
    dataloader = DataLoader(dataset)
    strategy = XLAStrategy()
    isinstance_return = True

    import torch_xla.distributed.parallel_loader as parallel_loader

    class MpDeviceLoaderMock(MagicMock):
        def __instancecheck__(self, instance):
            # to make `isinstance(dataloader, MpDeviceLoader)` pass with a mock as class
            return isinstance_return

    mp_loader_mock = MpDeviceLoaderMock()
    monkeypatch.setattr(parallel_loader, "MpDeviceLoader", mp_loader_mock)

    processed_dataloader = strategy.process_dataloader(dataloader)
    assert processed_dataloader is dataloader
    mp_loader_mock.assert_not_called()  # no-op

    isinstance_return = False
    processed_dataloader = strategy.process_dataloader(dataloader)
    mp_loader_mock.assert_called_with(dataloader, strategy.root_device)
    assert processed_dataloader.dataset == processed_dataloader._loader.dataset
    assert processed_dataloader.batch_sampler == processed_dataloader._loader.batch_sampler


def test_warning_if_tpus_not_used(tpu_available):
    with pytest.warns(UserWarning, match="TPU available but not used"):
        Trainer(accelerator="cpu")


@pytest.mark.parametrize(
    ("devices", "expected_device_ids"),
    [
        (1, [0]),
        (8, list(range(8))),
        ("8", list(range(8))),
        ([2], [2]),
        ("2,", [2]),
    ],
)
@RunIf(min_python="3.9")  # mocking issue
def test_trainer_config_device_ids(devices, expected_device_ids, tpu_available, monkeypatch):
    monkeypatch.setattr(lightning.fabric.accelerators.xla, "_using_pjrt", lambda: True)

    mock = DeviceMock()
    monkeypatch.setattr(torch, "device", mock)
    if _IS_WINDOWS:
        # simulate fork support on windows
        monkeypatch.setattr(torch.multiprocessing, "get_all_start_methods", lambda: ["fork", "spawn"])

    trainer = Trainer(accelerator="tpu", devices=devices)
    assert mock.mock_calls == [call("xla", i) for i in expected_device_ids]
    assert len(trainer.device_ids) == len(expected_device_ids)
    assert trainer.num_devices == len(expected_device_ids)
