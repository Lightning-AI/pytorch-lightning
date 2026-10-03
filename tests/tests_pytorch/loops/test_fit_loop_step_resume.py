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
from torch.utils.data import DataLoader, IterableDataset

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset


class _TraceModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.training_events = []
        self.validation_events = []
        self.sanity_events = []
        self.epoch_end_events = []

    def training_step(self, batch, batch_idx):
        self.training_events.append((self.current_epoch, self.global_step))
        return super().training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        events = self.sanity_events if self.trainer.sanity_checking else self.validation_events
        events.append((self.current_epoch, self.global_step, batch_idx))
        return super().validation_step(batch, batch_idx)

    def on_train_epoch_end(self):
        self.epoch_end_events.append((self.current_epoch, self.global_step))


class _IteratorTraceModel(_TraceModel):
    def training_step(self, dataloader_iter):
        first, _, _ = next(dataloader_iter)
        second, _, _ = next(dataloader_iter)
        self.training_events.append((self.current_epoch, self.global_step))
        return self(first).sum() + self(second).sum()


class _UnsizedDataset(IterableDataset):
    def __iter__(self):
        for _ in range(4):
            yield torch.randn(32)


@pytest.mark.parametrize("save_top_k", [0, -1])
@pytest.mark.parametrize("iteration_based", [False, True])
@pytest.mark.parametrize("loader_kind", ["standard", "iterator", "unsized", "accumulated"])
@pytest.mark.parametrize("with_validation", [False, True])
@pytest.mark.parametrize("reuse_model", [False, True])
def test_resume_completed_epoch_with_max_steps(
    tmp_path, iteration_based, loader_kind, with_validation, reuse_model, save_top_k
):
    use_iterator = loader_kind == "iterator"
    num_batches = 5 if loader_kind == "accumulated" else 4
    accumulation = 2 if loader_kind == "accumulated" else 1
    steps_per_epoch = (num_batches + accumulation - 1) // accumulation
    model_cls = _IteratorTraceModel if use_iterator else _TraceModel
    model = model_cls()
    dataset = (
        _UnsizedDataset() if loader_kind == "unsized" else RandomDataset(32, num_batches * (2 if use_iterator else 1))
    )
    train_loader = DataLoader(dataset)
    val_loader = DataLoader(RandomDataset(32, 2)) if with_validation else None

    def make_trainer(epochs):
        checkpoint = ModelCheckpoint(
            dirpath=tmp_path, save_last=True, save_top_k=save_top_k, save_on_train_epoch_end=True
        )
        trainer = Trainer(
            default_root_dir=tmp_path,
            accelerator="cpu",
            devices=1,
            max_epochs=-1 if iteration_based else epochs,
            max_steps=steps_per_epoch * epochs if iteration_based else -1,
            accumulate_grad_batches=accumulation,
            callbacks=[checkpoint],
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=2 if with_validation else 0,
            num_sanity_val_steps=2 if with_validation else 0,
        )
        return trainer, checkpoint

    trainer, checkpoint = make_trainer(1)
    trainer.fit(model, train_loader, val_loader)
    assert model.training_events == [(0, i // accumulation) for i in range(num_batches)]
    assert model.epoch_end_events == [(0, steps_per_epoch)]
    checkpoint_state = torch.load(checkpoint.last_model_path, weights_only=True)
    progress = checkpoint_state["loops"]["fit_loop"]["epoch_progress"]["current"]
    assert progress["processed"] == 1
    assert progress["completed"] == (1 if save_top_k == 0 else 0)
    assert checkpoint_state["loops"]["fit_loop"]["epoch_loop.batch_progress"]["is_last_batch"]

    model = model if reuse_model else model_cls()
    model.training_events.clear()
    model.validation_events.clear()
    model.sanity_events.clear()
    model.epoch_end_events.clear()
    resumed, _ = make_trainer(2)
    resumed.fit(model, train_loader, val_loader, ckpt_path=checkpoint.last_model_path)

    assert len(model.sanity_events) == (2 if with_validation and not iteration_based else 0)
    assert model.training_events == [(1, steps_per_epoch + i // accumulation) for i in range(num_batches)]
    assert model.validation_events == ([(1, 2 * steps_per_epoch, i) for i in range(2)] if with_validation else [])
    assert model.epoch_end_events == [(1, 2 * steps_per_epoch)]
    assert resumed.current_epoch == 2
    assert resumed.global_step == 2 * steps_per_epoch


@pytest.mark.parametrize("accumulate_grad_batches", [1, 2])
@pytest.mark.parametrize("with_validation", [False, True])
def test_resume_incomplete_epoch_with_max_steps(tmp_path, accumulate_grad_batches, with_validation):
    model = _TraceModel()
    train_loader = DataLoader(RandomDataset(32, 4))
    val_loader = DataLoader(RandomDataset(32, 2)) if with_validation else None

    def make_trainer(max_steps):
        checkpoint = ModelCheckpoint(dirpath=tmp_path, save_last=True, save_top_k=-1, save_on_train_epoch_end=True)
        trainer = Trainer(
            default_root_dir=tmp_path,
            accelerator="cpu",
            devices=1,
            max_epochs=-1,
            max_steps=max_steps,
            accumulate_grad_batches=accumulate_grad_batches,
            callbacks=[checkpoint],
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=2 if with_validation else 0,
            num_sanity_val_steps=0,
        )
        return trainer, checkpoint

    trainer, checkpoint = make_trainer(2 // accumulate_grad_batches)
    trainer.fit(model, train_loader, val_loader)
    checkpoint_state = torch.load(checkpoint.last_model_path, weights_only=True)
    assert checkpoint_state["loops"]["fit_loop"]["epoch_progress"]["current"]["completed"] == 0
    assert not checkpoint_state["loops"]["fit_loop"]["epoch_loop.batch_progress"]["is_last_batch"]
    model.training_events.clear()
    model.validation_events.clear()
    model.sanity_events.clear()
    model.epoch_end_events.clear()
    resumed, _ = make_trainer(4 // accumulate_grad_batches)
    resumed.fit(model, train_loader, val_loader, ckpt_path=checkpoint.last_model_path)

    assert model.training_events == [(0, i // accumulate_grad_batches) for i in range(2, 4)]
    assert model.validation_events == (
        [(0, 4 // accumulate_grad_batches, i) for i in range(2)] if with_validation else []
    )
    assert model.epoch_end_events == [(0, 4 // accumulate_grad_batches)]
    assert resumed.current_epoch == 1
    assert resumed.global_step == 4 // accumulate_grad_batches


@pytest.mark.parametrize("save_top_k", [0, -1])
def test_resume_completed_epoch_restores_stateful_loader(tmp_path, save_top_k):
    class StatefulChunks:
        def __init__(self):
            self.cursor = 0
            self.loaded_states = []

        def __len__(self):
            return 4

        def __iter__(self):
            for _ in range(len(self)):
                value = self.cursor
                self.cursor += 1
                yield torch.full((1, 32), float(value))

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.loaded_states.append(state)
            self.cursor = state["cursor"]

    class Model(_TraceModel):
        def __init__(self):
            super().__init__()
            self.batch_values = []

        def training_step(self, batch, batch_idx):
            self.batch_values.append(int(batch[0, 0].item()))
            return super().training_step(batch, batch_idx)

    def make_trainer(max_steps):
        checkpoint = ModelCheckpoint(
            dirpath=tmp_path, save_last=True, save_top_k=save_top_k, save_on_train_epoch_end=True
        )
        return Trainer(
            default_root_dir=tmp_path,
            accelerator="cpu",
            devices=1,
            max_epochs=-1,
            max_steps=max_steps,
            callbacks=[checkpoint],
            logger=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            limit_val_batches=0,
            num_sanity_val_steps=0,
        ), checkpoint

    trainer, checkpoint = make_trainer(4)
    trainer.fit(Model(), StatefulChunks())
    resumed, _ = make_trainer(8)
    model, loader = Model(), StatefulChunks()
    resumed.fit(model, loader, ckpt_path=checkpoint.last_model_path)

    assert loader.loaded_states == [{"cursor": 4}]
    assert model.batch_values == [4, 5, 6, 7]
    assert model.training_events == [(1, step) for step in range(4, 8)]
    assert resumed.current_epoch == 2
