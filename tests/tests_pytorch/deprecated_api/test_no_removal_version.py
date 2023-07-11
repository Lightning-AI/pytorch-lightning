import pytest

from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DDPStrategy


def test_configure_sharded_model():
    class MyModel(BoringModel):
        def configure_sharded_model(self) -> None:
            ...

    model = MyModel()
    trainer = Trainer(devices=1, accelerator="cpu", fast_dev_run=1)
    with pytest.deprecated_call(match="overridden `MyModel.configure_sharded_model"):
        trainer.fit(model)

    class MyModelBoth(MyModel):
        def configure_model(self):
            ...

    model = MyModelBoth()
    with pytest.raises(
        RuntimeError, match="Both `MyModelBoth.configure_model`, and `MyModelBoth.configure_sharded_model`"
    ):
        trainer.fit(model)


def test_ddp_is_distributed():
    strategy = DDPStrategy()
    with pytest.deprecated_call(match="is deprecated"):
        _ = strategy.is_distributed
