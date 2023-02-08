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
