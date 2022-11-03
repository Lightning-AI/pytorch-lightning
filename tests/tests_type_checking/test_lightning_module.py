from pathlib import Path

from pytorch_lightning import Trainer
from pytorch_lightning.demos.boring_classes import BoringModel


def test_load_from_checkpoint_type(tmp_path: Path) -> None:
    class MyModule(BoringModel):
        def __init__(self, some_parameter: int):
            super().__init__()
            self.save_hyperparameters()

        @property
        def parameter(self) -> int:
            return self.hparams.some_parameter

    net = MyModule(some_parameter=42)
    trainer = Trainer(default_root_dir=str(tmp_path), fast_dev_run=True)
    trainer.fit(net)
    checkpoint_path = str(tmp_path / "model.pt")
    trainer.save_checkpoint(checkpoint_path)

    net_loaded = MyModule.load_from_checkpoint(checkpoint_path)  # type: ignore
    assert net_loaded.parameter == 42
