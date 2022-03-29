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

import pytest
import torch

from pytorch_lightning import Callback, seed_everything, Trainer
from pytorch_lightning.accelerators import HPUAccelerator
from pytorch_lightning.strategies.hpu_parallel import HPUParallelStrategy
from pytorch_lightning.strategies.single_hpu import SingleHPUStrategy
from pytorch_lightning.utilities import _HPU_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from tests.helpers.boring_model import BoringModel
from tests.helpers.datamodules import ClassifDataModule
from tests.helpers.runif import RunIf
from tests.helpers.simple_models import ClassificationModel


@RunIf(hpu=True)
def test_availability():
    assert HPUAccelerator.is_available()


@pytest.mark.skipif(_HPU_AVAILABLE, reason="test requires non-HPU machine")
def test_fail_if_no_hpus():
    with pytest.raises(MisconfigurationException, match="HPUAccelerator can not run on your system"):
        Trainer(accelerator="hpu", devices=1)


@RunIf(hpu=True)
def test_accelerator_selected():
    trainer = Trainer(accelerator="hpu")
    assert isinstance(trainer.accelerator, HPUAccelerator)


@RunIf(hpu=True)
def test_all_stages(tmpdir, hpus):
    """Tests all the model stages using BoringModel on HPU."""
    model = BoringModel()

    trainer = Trainer(
        default_root_dir=tmpdir,
        fast_dev_run=True,
        accelerator="hpu",
        devices=hpus,
        precision=16,
    )
    trainer.fit(model)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


@RunIf(hpu=True)
def test_optimization(tmpdir):
    seed_everything(42)

    dm = ClassifDataModule(length=1024)
    model = ClassificationModel()

    trainer = Trainer(default_root_dir=tmpdir, max_epochs=1, accelerator="hpu", devices=1)

    # fit model
    trainer.fit(model, dm)
    assert trainer.state.finished, f"Training failed with {trainer.state}"
    assert dm.trainer is not None

    # validate
    result = trainer.validate(datamodule=dm)
    assert dm.trainer is not None
    assert result[0]["val_acc"] > 0.7

    # test
    result = trainer.test(model, datamodule=dm)
    assert dm.trainer is not None
    test_result = result[0]["test_acc"]
    assert test_result > 0.6

    # test saved model
    model_path = os.path.join(tmpdir, "model.pt")
    trainer.save_checkpoint(model_path)

    model = ClassificationModel.load_from_checkpoint(model_path)

    trainer = Trainer(default_root_dir=tmpdir, accelerator="hpu", devices=1)

    result = trainer.test(model, datamodule=dm)
    saved_result = result[0]["test_acc"]
    assert saved_result == test_result


@RunIf(hpu=True)
def test_stages_correct(tmpdir):
    """Ensure all stages correctly are traced correctly by asserting the output for each stage."""

    class StageModel(BoringModel):
        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            loss = loss.get("loss")
            # tracing requires a loss value that depends on the model.
            # force it to be a value but ensure we use the loss.
            loss = (loss - loss) + torch.tensor(1)
            return {"loss": loss}

        def validation_step(self, batch, batch_idx):
            loss = super().validation_step(batch, batch_idx)
            x = loss.get("x")
            x = (x - x) + torch.tensor(2)
            return {"x": x}

        def test_step(self, batch, batch_idx):
            loss = super().test_step(batch, batch_idx)
            y = loss.get("y")
            y = (y - y) + torch.tensor(3)
            return {"y": y}

        def predict_step(self, batch, batch_idx, dataloader_idx=None):
            output = super().predict_step(batch, batch_idx)
            return (output - output) + torch.tensor(4)

    class TestCallback(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
            assert outputs["loss"].item() == 1

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
            assert outputs["x"].item() == 2

        def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
            assert outputs["y"].item() == 3

        def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx) -> None:
            assert torch.all(outputs == 4).item()

    model = StageModel()
    trainer = Trainer(
        default_root_dir=tmpdir, fast_dev_run=True, accelerator="hpu", devices=1, callbacks=TestCallback()
    )
    trainer.fit(model)
    trainer.test(model)
    trainer.validate(model)
    trainer.predict(model)


@RunIf(hpu=True)
def test_accelerator_hpu():

    trainer = Trainer(accelerator="hpu", devices=1)
    assert isinstance(trainer.accelerator, HPUAccelerator)
    assert trainer.num_devices == 1

    trainer = Trainer(accelerator="hpu")
    assert isinstance(trainer.accelerator, HPUAccelerator)
    assert trainer.num_devices == 8

    trainer = Trainer(accelerator="auto", devices=8)
    assert isinstance(trainer.accelerator, HPUAccelerator)
    assert trainer.num_devices == 8


@RunIf(hpu=True)
def test_accelerator_hpu_with_single_device():

    trainer = Trainer(accelerator="hpu", devices=1)

    assert isinstance(trainer.strategy, SingleHPUStrategy)
    assert isinstance(trainer.accelerator, HPUAccelerator)


@RunIf(hpu=True)
def test_accelerator_hpu_with_multiple_devices():

    trainer = Trainer(accelerator="hpu", devices=8)

    assert isinstance(trainer.strategy, HPUParallelStrategy)
    assert isinstance(trainer.accelerator, HPUAccelerator)


@RunIf(hpu=True)
def test_accelerator_auto_with_devices_hpu():

    trainer = Trainer(accelerator="auto", devices=8)

    assert isinstance(trainer.strategy, HPUParallelStrategy)


@RunIf(hpu=True)
def test_strategy_choice_hpu_plugin():
    trainer = Trainer(strategy=SingleHPUStrategy(device=torch.device("hpu")), accelerator="hpu", devices=1)
    assert isinstance(trainer.strategy, SingleHPUStrategy)

    trainer = Trainer(accelerator="hpu", devices=1)
    assert isinstance(trainer.strategy, SingleHPUStrategy)


@RunIf(hpu=True)
def test_strategy_choice_hpu_parallel_plugin():
    trainer = Trainer(
        strategy=HPUParallelStrategy(parallel_devices=[torch.device("hpu")] * 8), accelerator="hpu", devices=8
    )
    assert isinstance(trainer.strategy, HPUParallelStrategy)

    trainer = Trainer(accelerator="hpu", devices=8)
    assert isinstance(trainer.strategy, HPUParallelStrategy)


@RunIf(hpu=True)
def test_devices_auto_choice_hpu():
    trainer = Trainer(accelerator="auto", devices="auto")
    assert trainer.num_devices == 8


@RunIf(hpu=True)
@pytest.mark.parametrize("hpus", [1])
def test_inference_only(tmpdir, hpus):
    model = BoringModel()

    trainer = Trainer(default_root_dir=tmpdir, fast_dev_run=True, accelerator="hpu", devices=hpus)
    trainer.validate(model)
    trainer.test(model)
    trainer.predict(model)


def test_hpu_auto_device_count():
    assert HPUAccelerator.auto_device_count() == 8


@RunIf(hpu=True)
def test_hpu_unsupported_device_type():
    with pytest.raises(MisconfigurationException, match="`devices` for `HPUAccelerator` must be int, string or None."):
        Trainer(accelerator="hpu", devices=[1])
