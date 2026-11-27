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
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch import LightningModule, Trainer


class _FailingEvaluationModel(LightningModule):
    def __init__(self, failure):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(2, 2, bias=False),
            torch.nn.BatchNorm1d(2),
            torch.nn.Dropout(0.25),
        )
        self.network[1].eval()
        self.network[1].requires_grad_(False)
        self.failure = failure
        self.error = RuntimeError("intentional evaluation failure")

    def _fail(self, point):
        if self.failure == point:
            raise self.error

    def _evaluate(self, batch):
        assert not self.training
        output = self.network(batch[0])
        self._fail("step")
        return output

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch)

    def test_step(self, batch, batch_idx):
        return self._evaluate(batch)

    def predict_step(self, batch, batch_idx):
        return self._evaluate(batch)

    def on_validation_start(self):
        self._fail("start")

    def on_test_start(self):
        self._fail("start")

    def on_predict_start(self):
        self._fail("start")

    def on_validation_epoch_end(self):
        self._fail("end")

    def on_test_epoch_end(self):
        self._fail("end")

    def on_predict_epoch_end(self):
        self._fail("end")

    def training_step(self, batch, batch_idx):
        assert self.training
        assert self.network[-1].training
        assert not self.network[1].training
        return self.network(batch[0]).square().mean()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1)


def _modes(model):
    return {name: module.training for name, module in model.named_modules()}


def _trainer(tmp_path, inference_mode=True):
    return Trainer(
        default_root_dir=tmp_path,
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        inference_mode=inference_mode,
    )


@pytest.mark.parametrize("stage", ["validate", "test", "predict"])
@pytest.mark.parametrize("inference_mode", [False, True])
@pytest.mark.parametrize("failure", [None, "start", "step", "end"])
def test_failed_evaluation_restores_module_modes(tmp_path, stage, inference_mode, failure):
    torch.manual_seed(1337)
    model = _FailingEvaluationModel(failure)
    expected_modes = _modes(model)
    loader = DataLoader(TensorDataset(torch.arange(16).float().reshape(8, 2) / 10), batch_size=4)
    trainer = _trainer(tmp_path, inference_mode)
    trainer.limit_val_batches = 1.0
    if failure is None:
        getattr(trainer, stage)(model, dataloaders=loader)
    else:
        with pytest.raises(RuntimeError, match="intentional evaluation failure") as exc:
            getattr(trainer, stage)(model, dataloaders=loader)
        assert exc.value is model.error
    assert _modes(model) == expected_modes

    model.failure = None
    getattr(trainer, stage)(model, dataloaders=loader)
    assert _modes(model) == expected_modes
    before = model.network[0].weight.detach().clone()
    running_mean = model.network[1].running_mean.clone()
    trainer.limit_val_batches = 0
    trainer.fit(model, train_dataloaders=loader)
    assert not torch.equal(model.network[0].weight, before)
    torch.testing.assert_close(model.network[1].running_mean, running_mean)


@pytest.mark.parametrize("stage", ["validate", "test", "predict"])
def test_mode_restore_does_not_reuse_another_models_snapshot(tmp_path, stage):
    trainer = _trainer(tmp_path)
    trainer.limit_val_batches = 1.0
    loader = DataLoader(TensorDataset(torch.ones(8, 2)), batch_size=4)
    first = _FailingEvaluationModel(None).eval()
    getattr(trainer, stage)(first, dataloaders=loader)
    second = _FailingEvaluationModel(None)
    expected_modes = _modes(second)
    with pytest.raises(RuntimeError, match="no.*dataloader_idx"):
        getattr(trainer, stage)(second, dataloaders=[loader, loader])
    assert _modes(second) == expected_modes
