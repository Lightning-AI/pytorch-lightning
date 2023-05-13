import torch

from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.demos.boring_classes import BoringModel, RandomDataset, RandomIterableDataset
from lightning.pytorch.strategies import DeepSpeedStrategy


class ModelParallelBoringModel(BoringModel):
    def __init__(self):
        super().__init__()
        self.layer = None

    def configure_sharded_model(self) -> None:
        self.layer = torch.nn.Linear(32, 2)

    def on_load_checkpoint(self, checkpoint) -> None:
        self.configure_sharded_model()

    
def main():
    model = ModelParallelBoringModel()
    trainer = Trainer(
        strategy=DeepSpeedStrategy(stage=3),
        accelerator="cuda",
        devices=2,
        fast_dev_run=True,
        precision="16-mixed",
    )
    trainer.test(model)
    print(model.ds_inflight_param_registry)
    trainer.fit(model)


if __name__ == "__main__":
    main()