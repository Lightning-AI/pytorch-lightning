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
from copy import deepcopy

import pytest
import torch
from torch.utils.data import DataLoader

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EMAWeightAveraging, ModelCheckpoint, WeightAveraging


class _ScalarModel(LightningModule):
    def __init__(self, momentum=0.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.register_buffer("batches", torch.tensor(0.0))
        self.momentum = momentum

    def training_step(self, batch, batch_idx):
        self.batches.add_(1)
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("weight", self.weight)

    def train_dataloader(self):
        return DataLoader([torch.tensor(0.0)])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.1, momentum=self.momentum)


def _averaging(kind, use_buffers=True):
    if kind == "ema":
        return EMAWeightAveraging(decay=0.8, use_buffers=use_buffers)
    return WeightAveraging(use_buffers=use_buffers)


def _trainer(folder, averaging, epochs, mode):
    checkpoint = ModelCheckpoint(dirpath=folder, save_top_k=-1 if mode == "epoch" else 0, save_last=True)
    trainer = Trainer(
        default_root_dir=folder,
        max_epochs=epochs,
        callbacks=[averaging, checkpoint],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    return trainer, checkpoint


def _save_path(trainer, checkpoint, folder, mode):
    if mode == "manual":
        path = folder / "manual.ckpt"
        trainer.save_checkpoint(path)
        return path
    return checkpoint.last_model_path


@pytest.mark.parametrize("kind", ["swa", "ema"])
@pytest.mark.parametrize("mode", ["epoch", "final", "manual"])
@pytest.mark.parametrize("use_buffers", [False, True])
def test_final_checkpoint_keeps_current_and_averaged_weights(tmp_path, kind, mode, use_buffers):
    model = _ScalarModel()
    averaging = _averaging(kind, use_buffers)
    trainer, checkpoint = _trainer(tmp_path, averaging, 2, mode)
    trainer.fit(model)
    path = _save_path(trainer, checkpoint, tmp_path, mode)
    saved = torch.load(path, weights_only=False)

    expected_average = 0.72 if kind == "swa" else 0.768
    torch.testing.assert_close(model.weight, torch.tensor(expected_average))
    torch.testing.assert_close(saved["state_dict"]["weight"], torch.tensor(expected_average))
    torch.testing.assert_close(saved["current_model_state"]["weight"], torch.tensor(0.64))
    torch.testing.assert_close(saved["current_model_state"]["batches"], torch.tensor(2.0))
    expected_batches = (1.5 if kind == "swa" else 1.2) if use_buffers else 2.0
    torch.testing.assert_close(saved["state_dict"]["batches"], torch.tensor(expected_batches))
    assert saved["averaging_state"]["n_averaged"] == 2


@pytest.mark.parametrize("kind", ["swa", "ema"])
@pytest.mark.parametrize("mode", ["final", "manual"])
@pytest.mark.parametrize("momentum", [0.0, 0.8])
def test_resume_after_completed_fit_matches_uninterrupted_training(tmp_path, kind, mode, momentum):
    full_model = _ScalarModel(momentum)
    full_average = _averaging(kind)
    full_trainer, _ = _trainer(tmp_path / "full", full_average, 4, mode)
    full_trainer.fit(full_model)

    partial_model = _ScalarModel(momentum)
    partial_average = _averaging(kind)
    partial_trainer, checkpoint = _trainer(tmp_path / "partial", partial_average, 2, mode)
    partial_trainer.fit(partial_model)
    path = _save_path(partial_trainer, checkpoint, tmp_path / "partial", mode)

    resumed_model = _ScalarModel(momentum)
    resumed_average = _averaging(kind)
    resumed_trainer, _ = _trainer(tmp_path / "resumed", resumed_average, 4, mode)
    resumed_trainer.fit(resumed_model, ckpt_path=path)

    for name, expected in full_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], expected, rtol=0, atol=0)
    assert resumed_average._average_model.n_averaged == full_average._average_model.n_averaged == 4
    assert resumed_trainer.global_step == full_trainer.global_step == 4
    if momentum:
        torch.testing.assert_close(
            resumed_trainer.optimizers[0].state[resumed_model.weight]["momentum_buffer"],
            full_trainer.optimizers[0].state[full_model.weight]["momentum_buffer"],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("kind", ["swa", "ema"])
def test_validation_after_fit_keeps_final_average_and_resumable_checkpoint(tmp_path, kind):
    model = _ScalarModel()
    averaging = _averaging(kind)
    trainer, checkpoint = _trainer(tmp_path, averaging, 2, "final")
    trainer.fit(model)
    expected = model.weight.detach().clone()
    for checkpoint_path in (None, checkpoint.last_model_path):
        result = trainer.validate(model, dataloaders=DataLoader([torch.tensor(0.0)]), ckpt_path=checkpoint_path)
        assert result[0]["weight"] == expected.item()
        if checkpoint_path is None:
            torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
    path = _save_path(trainer, checkpoint, tmp_path, "manual")
    saved = torch.load(path, weights_only=False)
    torch.testing.assert_close(saved["state_dict"]["weight"], expected)
    torch.testing.assert_close(saved["current_model_state"]["weight"], torch.tensor(0.64))


@pytest.mark.parametrize("kind", ["swa", "ema"])
def test_loading_checkpoint_after_finalization_restores_callback_roles(tmp_path, kind):
    model = _ScalarModel()
    averaging = _averaging(kind)
    trainer, checkpoint = _trainer(tmp_path, averaging, 2, "epoch")
    trainer.fit(model)
    saved = torch.load(checkpoint.last_model_path, weights_only=False)
    # Reload into the same native callback after on_train_end finalized the model.
    averaging.on_load_checkpoint(trainer, model, deepcopy(saved))
    round_trip = {"state_dict": model.state_dict()}
    averaging.on_save_checkpoint(trainer, model, round_trip)
    for slot in ("state_dict", "current_model_state"):
        for name, value in saved[slot].items():
            torch.testing.assert_close(round_trip[slot][name], value, rtol=0, atol=0)
