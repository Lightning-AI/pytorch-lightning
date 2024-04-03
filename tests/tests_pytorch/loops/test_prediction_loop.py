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
import itertools

import pytest
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset
from lightning.pytorch.overrides.distributed import _IndexBatchSamplerWrapper
from torch.utils.data import DataLoader, DistributedSampler, SequentialSampler


def test_prediction_loop_stores_predictions(tmp_path):
    class MyModel(BoringModel):
        def predict_step(self, batch, batch_idx):
            return batch_idx

    model = MyModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_predict_batches=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predictions = trainer.predict(model, return_predictions=True)
    assert predictions == [0, 1]
    # the predictions are still available
    assert trainer.predict_loop.predictions == predictions

    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_predict_batches=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    predictions = trainer.predict(model, return_predictions=False)
    assert predictions is None
    assert trainer.predict_loop.predictions == []


@pytest.mark.parametrize("use_distributed_sampler", [False, True])
def test_prediction_loop_batch_sampler_set_epoch_called(tmp_path, use_distributed_sampler):
    """Tests that set_epoch is called on the dataloader's batch sampler (if any) during prediction."""
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_predict_batches=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        strategy="ddp",
        devices=1,
        accelerator="cpu",
        use_distributed_sampler=use_distributed_sampler,
    )

    class MyModel(BoringModel):
        def predict_dataloader(self):
            dataset = RandomDataset(32, 64)
            sampler = None
            if not use_distributed_sampler:
                sampler = DistributedSampler(dataset)
            return DataLoader(dataset, sampler=sampler)

    model = MyModel()
    trainer.fit_loop.epoch_progress.current.processed = 2
    trainer.predict(model)

    # torch will set this .sampler attribute for backwards compatibility, but in reality, the batch sampler is used
    assert isinstance(trainer.predict_dataloaders.sampler, SequentialSampler)
    batch_sampler = trainer.predict_dataloaders.batch_sampler
    assert isinstance(batch_sampler, _IndexBatchSamplerWrapper)
    assert isinstance(batch_sampler.sampler, DistributedSampler)
    assert batch_sampler.sampler.epoch == 2


def test_prediction_loop_with_iterable_dataset(tmp_path):
    class MyModel(BoringModel):
        def predict_step(self, batch, batch_idx, dataloader_idx=0):
            return (batch, batch_idx, dataloader_idx)

    model = MyModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_predict_batches=3,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
        devices=1,
    )
    preds = trainer.predict(model, itertools.count())
    assert preds == [(0, 0, 0), (1, 1, 0), (2, 2, 0)]

    preds = trainer.predict(model, [itertools.count(), itertools.count()])
    assert preds == [[(0, 0, 0), (1, 1, 0), (2, 2, 0)], [(0, 0, 1), (1, 1, 1), (2, 2, 1)]]

    preds = trainer.predict(model, {"a": [0, 1], "b": [2, 3]})
    assert preds == [[(0, 0, 0), (1, 1, 0)], [(2, 0, 1), (3, 1, 1)]]

    preds = trainer.predict(model, [[0, 1], [2, 3]])
    assert preds == [[(0, 0, 0), (1, 1, 0)], [(2, 0, 1), (3, 1, 1)]]

    class MyModel(LightningModule):
        batch_start_ins = []
        step_outs = []
        batch_end_ins = []

        def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
            self.batch_start_ins.append((batch, batch_idx, dataloader_idx))

        def predict_step(self, dataloader_iter):
            self.step_outs.append(next(dataloader_iter))

        def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
            self.batch_end_ins.append((batch, batch_idx, dataloader_idx))

    model = MyModel()
    trainer.predict(model, {"a": [0, 1], "b": [2, 3]})

    assert model.batch_start_ins == [(None, 0, 0)] + model.step_outs[:-1]
    assert model.step_outs == [(0, 0, 0), (1, 1, 0), (2, 0, 1), (3, 1, 1)]
    assert model.batch_end_ins == model.step_outs


def test_invalid_dataloader_idx_raises_step(tmp_path):
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)

    class ExtraDataloaderIdx(BoringModel):
        def predict_step(self, batch, batch_idx, dataloader_idx): ...

    model = ExtraDataloaderIdx()
    with pytest.raises(RuntimeError, match="have included `dataloader_idx` in `ExtraDataloaderIdx.predict_step"):
        trainer.predict(model)

    class GoodDefault(BoringModel):
        def predict_step(self, batch, batch_idx, dataloader_idx=0): ...

    model = GoodDefault()
    trainer.predict(model)

    class ExtraDlIdxOtherName(BoringModel):
        def predict_step(self, batch, batch_idx, dl_idx): ...

    model = ExtraDlIdxOtherName()
    # different names are not supported
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.predict(model)

    class MultipleDataloader(BoringModel):
        def predict_step(self, batch, batch_idx): ...

        def predict_dataloader(self):
            return [super().predict_dataloader(), super().predict_dataloader()]

    model = MultipleDataloader()
    with pytest.raises(RuntimeError, match="no `dataloader_idx` argument in `MultipleDataloader.predict_step"):
        trainer.predict(model)

    class IgnoringModel(MultipleDataloader):
        def predict_step(self, batch, batch_idx, *_): ...

    model = IgnoringModel()
    trainer.predict(model)

    class IgnoringModel2(MultipleDataloader):
        def predict_step(self, batch, batch_idx, **_): ...

    model = IgnoringModel2()
    with pytest.raises(RuntimeError, match="no `dataloader_idx` argument in `IgnoringModel2.predict_step"):
        trainer.predict(model)


def test_invalid_dataloader_idx_raises_batch_start(tmp_path):
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)

    class ExtraDataloaderIdx(BoringModel):
        def on_predict_batch_start(self, batch, batch_idx, dataloader_idx): ...

    model = ExtraDataloaderIdx()
    with pytest.raises(
        RuntimeError, match="have included `dataloader_idx` in `ExtraDataloaderIdx.on_predict_batch_start"
    ):
        trainer.predict(model)

    class GoodDefault(BoringModel):
        def on_predict_batch_start(self, batch, batch_idx, dataloader_idx=0): ...

    model = GoodDefault()
    trainer.predict(model)

    class ExtraDlIdxOtherName(BoringModel):
        def on_predict_batch_start(self, batch, batch_idx, dl_idx): ...

    model = ExtraDlIdxOtherName()
    # different names are not supported
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.predict(model)

    class MultipleDataloader(BoringModel):
        def on_predict_batch_start(self, batch, batch_idx): ...

        def predict_dataloader(self):
            return [super().predict_dataloader(), super().predict_dataloader()]

    model = MultipleDataloader()
    with pytest.raises(
        RuntimeError, match="no `dataloader_idx` argument in `MultipleDataloader.on_predict_batch_start"
    ):
        trainer.predict(model)

    class IgnoringModel(MultipleDataloader):
        def on_predict_batch_start(self, batch, batch_idx, *_): ...

    model = IgnoringModel()
    trainer.predict(model)

    class IgnoringModel2(MultipleDataloader):
        def on_predict_batch_start(self, batch, batch_idx, **_): ...

    model = IgnoringModel2()
    with pytest.raises(RuntimeError, match="no `dataloader_idx` argument in `IgnoringModel2.on_predict_batch_start"):
        trainer.predict(model)


def test_invalid_dataloader_idx_raises_batch_end(tmp_path):
    trainer = Trainer(default_root_dir=tmp_path, fast_dev_run=True)

    class ExtraDataloaderIdx(BoringModel):
        def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx): ...

    model = ExtraDataloaderIdx()
    with pytest.raises(
        RuntimeError, match="have included `dataloader_idx` in `ExtraDataloaderIdx.on_predict_batch_end"
    ):
        trainer.predict(model)

    class GoodDefault(BoringModel):
        def on_predict_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0): ...

    model = GoodDefault()
    trainer.predict(model)

    class ExtraDlIdxOtherName(BoringModel):
        def on_predict_batch_end(self, outputs, batch, batch_idx, dl_idx): ...

    model = ExtraDlIdxOtherName()
    # different names are not supported
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'dl_idx"):
        trainer.predict(model)

    class MultipleDataloader(BoringModel):
        def on_predict_batch_end(self, outputs, batch, batch_idx): ...

        def predict_dataloader(self):
            return [super().predict_dataloader(), super().predict_dataloader()]

    model = MultipleDataloader()
    with pytest.raises(RuntimeError, match="no `dataloader_idx` argument in `MultipleDataloader.on_predict_batch_end"):
        trainer.predict(model)

    class IgnoringModel(MultipleDataloader):
        def on_predict_batch_end(self, outputs, batch, batch_idx, *_): ...

    model = IgnoringModel()
    trainer.predict(model)

    class IgnoringModel2(MultipleDataloader):
        def on_predict_batch_end(self, outputs, batch, batch_idx, **_): ...

    model = IgnoringModel2()
    with pytest.raises(RuntimeError, match="no `dataloader_idx` argument in `IgnoringModel2.on_predict_batch_end"):
        trainer.predict(model)


def test_prediction_loop_when_batch_idx_argument_is_not_given(tmp_path):
    class TestModel(BoringModel):
        def __init__(self) -> None:
            super().__init__()
            self.predict_step_called = False

        def predict_step(self, batch):
            self.predict_step_called = True
            return self.step(batch)

    trainer = Trainer(
        default_root_dir=tmp_path,
        fast_dev_run=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    model = TestModel()

    trainer.predict(model)
    assert model.predict_step_called
