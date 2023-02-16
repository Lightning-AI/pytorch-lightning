import itertools
from unittest import mock
from unittest.mock import call

import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel


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


def test_prediction_loop_batch_sampler_set_epoch_called(tmp_path):
    """Tests that set_epoch is called on the dataloader's batch sampler (if any) during prediction."""
    model = BoringModel()
    trainer = Trainer(
        default_root_dir=tmp_path,
        limit_predict_batches=1,
        enable_model_summary=False,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit_loop.epoch_progress.current.processed = 2

    with mock.patch("lightning.pytorch.overrides.distributed.IndexBatchSamplerWrapper.set_epoch") as set_epoch_mock:
        trainer.predict(model)
    assert set_epoch_mock.mock_calls == [call(2)]


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
    )
    preds = trainer.predict(model, itertools.count())
    assert preds == [(0, 0, 0), (1, 1, 0), (2, 2, 0)]

    preds = trainer.predict(model, [itertools.count(), itertools.count()])
    assert preds == [[(0, 0, 0), (1, 1, 0), (2, 2, 0)], [(0, 0, 1), (1, 1, 1), (2, 2, 1)]]

    # TODO(carmocca): this shouldn't raise
    with pytest.raises(ValueError, match="Mismatch in number of limits"):
        trainer.predict(model, {"a": [0, 1], "b": [2, 3]})
    with pytest.raises(ValueError, match="Mismatch in number of limits"):
        trainer.predict(model, [0, 1, 2])

    class MyModel(BoringModel):
        def predict_step(self, dataloader_iter, batch_idx, dataloader_idx=0):
            ...

    model = MyModel()
    with pytest.raises(NotImplementedError, match="dataloader_iter.*is not supported with multiple dataloaders"):
        trainer.predict(model, {"a": [0, 1], "b": [2, 3]})
