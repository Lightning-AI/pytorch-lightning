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
# limitations under the License
import collections
from copy import deepcopy

import pytest
import torch
from torch import nn

from pytorch_lightning import Trainer
from pytorch_lightning.accelerators.cpu import CPUAccelerator
from pytorch_lightning.accelerators.tpu import TPUAccelerator
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.plugins import TPUSpawnPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.runif import RunIf
from tests.helpers.utils import pl_multi_process_test


class WeightSharingModule(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(32, 10, bias=False)
        self.layer_2 = nn.Linear(10, 32, bias=False)
        self.layer_3 = nn.Linear(32, 10, bias=False)
        self.layer_3.weight = self.layer_1.weight

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        return x


@RunIf(tpu=True)
@pl_multi_process_test
def test_resume_training_on_cpu(tmpdir):
    """Checks if training can be resumed from a saved checkpoint on CPU"""
    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(checkpoint_callback=True, max_epochs=1, tpu_cores=8)
    trainer.fit(model)

    model_path = trainer.checkpoint_callback.best_model_path

    # Verify saved Tensors are on CPU
    ckpt = torch.load(model_path)
    weight_tensor = list(ckpt["state_dict"].values())[0]
    assert weight_tensor.device == torch.device("cpu")

    # Verify that training is resumed on CPU
    trainer = Trainer(
        resume_from_checkpoint=model_path, checkpoint_callback=True, max_epochs=1, default_root_dir=tmpdir
    )
    trainer.fit(model)
    assert trainer.state.finished, f"Training failed with {trainer.state}"


@RunIf(tpu=True)
@pl_multi_process_test
def test_if_test_works_after_train(tmpdir):
    """Ensure that .test() works after .fit()"""

    # Train a model on TPU
    model = BoringModel()
    trainer = Trainer(max_epochs=1, tpu_cores=8, default_root_dir=tmpdir, fast_dev_run=True)
    trainer.fit(model)
    assert len(trainer.test(model)) == 1


@RunIf(tpu=True)
@pl_multi_process_test
def test_weight_tying_warning(tmpdir, capsys=None):
    """
    Ensure a warning is thrown if model parameter lengths do not match
    post moving to device.
    """

    model = WeightSharingModule()
    trainer = Trainer(checkpoint_callback=True, max_epochs=1, tpu_cores=1)

    with pytest.warns(UserWarning, match=r"The model layers do not match after moving to the target device."):
        trainer.fit(model)


@RunIf(tpu=True)
@pl_multi_process_test
def test_if_weights_tied(tmpdir, capsys=None):
    """
    Test if weights are properly tied on `on_post_move_to_device`.
    Ensure no warning for parameter mismatch is thrown.
    """

    class Model(WeightSharingModule):
        def on_post_move_to_device(self):
            self.layer_3.weight = self.layer_1.weight

    model = Model()
    trainer = Trainer(checkpoint_callback=True, max_epochs=1, tpu_cores=1)

    with pytest.warns(UserWarning, match="The model layers do not match"):
        trainer.fit(model)


@RunIf(tpu=True)
def test_accelerator_tpu():

    trainer = Trainer(accelerator="tpu", tpu_cores=8)

    assert trainer._device_type == "tpu"
    assert isinstance(trainer.accelerator, TPUAccelerator)

    with pytest.raises(
        MisconfigurationException, match="You passed `accelerator='tpu'`, but you didn't pass `tpu_cores` to `Trainer`"
    ):
        trainer = Trainer(accelerator="tpu")


@RunIf(tpu=True)
def test_accelerator_cpu_with_tpu_cores_flag():

    trainer = Trainer(accelerator="cpu", tpu_cores=8)

    assert trainer._device_type == "cpu"
    assert isinstance(trainer.accelerator, CPUAccelerator)


@RunIf(tpu=True)
def test_accelerator_tpu_with_auto():

    trainer = Trainer(accelerator="auto", tpu_cores=8)

    assert trainer._device_type == "tpu"
    assert isinstance(trainer.accelerator, TPUAccelerator)


@RunIf(tpu=True)
def test_accelerator_tpu_with_devices():

    trainer = Trainer(accelerator="tpu", devices=8)

    assert trainer.tpu_cores == 8
    assert isinstance(trainer.training_type_plugin, TPUSpawnPlugin)
    assert isinstance(trainer.accelerator, TPUAccelerator)


@RunIf(tpu=True)
def test_accelerator_auto_with_devices_tpu():

    trainer = Trainer(accelerator="auto", devices=8)

    assert trainer._device_type == "tpu"
    assert trainer.tpu_cores == 8


@RunIf(tpu=True)
def test_accelerator_tpu_with_tpu_cores_priority():
    """Test for checking `tpu_cores` flag takes priority over `devices`."""

    tpu_cores = 8
    with pytest.warns(UserWarning, match="The flag `devices=1` will be ignored,"):
        trainer = Trainer(accelerator="tpu", devices=1, tpu_cores=tpu_cores)

    assert trainer.tpu_cores == tpu_cores


@RunIf(tpu=True)
def test_set_devices_if_none_tpu():

    trainer = Trainer(accelerator="tpu", tpu_cores=8)
    assert trainer.devices == 8


@RunIf(tpu=True)
def test_manual_optimization_tpus(tmpdir):
    class ManualOptimizationModel(BoringModel):

        count = 0
        called = collections.defaultdict(int)

        def __init__(self):
            super().__init__()
            self.automatic_optimization = False

        @property
        def should_update(self):
            return self.count % 2 == 0

        def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
            self.called["on_train_batch_start"] += 1
            self.weight_before = self.layer.weight.clone()

        def training_step(self, batch, batch_idx):
            self.called["training_step"] += 1
            opt = self.optimizers()
            output = self.layer(batch)
            loss = self.loss(batch, output)

            if self.should_update:
                self.manual_backward(loss)
                opt.step()
                opt.zero_grad()
            return loss

        def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.called["on_train_batch_end"] += 1
            after_before = self.layer.weight.clone()
            if self.should_update:
                assert not torch.equal(self.weight_before, after_before), self.count
            else:
                assert torch.equal(self.weight_before, after_before)
            assert torch.all(self.layer.weight.grad == 0)
            self.count += 1

        def on_train_end(self):
            assert self.called["training_step"] == 5
            assert self.called["on_train_batch_start"] == 5
            assert self.called["on_train_batch_end"] == 5

    class TestManualOptimizationCallack(Callback):
        def on_train_end(self, trainer, pl_module):

            opt = pl_module.optimizers()
            assert opt._total_optimizer_step_calls == 3

    model = ManualOptimizationModel()
    model_copy = deepcopy(model)
    model.training_step_end = None
    model.training_epoch_end = None

    trainer = Trainer(
        max_epochs=1,
        default_root_dir=tmpdir,
        limit_train_batches=5,
        limit_test_batches=0,
        limit_val_batches=0,
        tpu_cores=8,
        callbacks=[TestManualOptimizationCallack()],
    )
    trainer.fit(model)

    for param, param_copy in zip(model.parameters(), model_copy.parameters()):
        assert not torch.equal(param.cpu().data, param_copy.data)


@RunIf(tpu=True)
def test_ddp_cpu_not_supported_on_tpus():

    with pytest.raises(MisconfigurationException, match="`accelerator='ddp_cpu'` is not supported on TPU machines"):
        Trainer(accelerator="ddp_cpu")
