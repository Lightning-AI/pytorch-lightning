import pytest
from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from tests.helpers import BoringModel, BoringDataModule
from torch.utils.data import DataLoader


class BatchSizeDataModule(BoringDataModule):

    def __init__(self, batch_size=2):
        super().__init__()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.random_train, batch_size=self.batch_size)


class BatchSizeModel(BoringModel):

    def __init__(self, batch_size=2):
        super().__init__()
        self.save_hyperparameters()


# @RunIf()
@pytest.mark.parametrize("model,datamodule", [
    (BatchSizeModel(2), None),
    (BatchSizeModel(2), BatchSizeDataModule(2))
])
def test_scale_batch_size_method_with_model_or_datamodule(tmpdir, model, datamodule):
    """ Test the tuner method `Tuner.scale_batch_size` with a datamodule. """
    trainer = Trainer(
        default_root_dir=tmpdir,
        gpus=1,
        limit_train_batches=1,
        limit_val_batches=0,
        max_epochs=1,
    )
    tuner = Tuner(trainer)
    new_batch_size = tuner.scale_batch_size(
        model=model,
        mode="binsearch",
        init_val=4,
        max_trials=2,
        datamodule=datamodule
    )
    assert new_batch_size == 16
    assert model.hparams.batch_size == 16
    if datamodule is not None:
        assert datamodule.batch_size == 16
