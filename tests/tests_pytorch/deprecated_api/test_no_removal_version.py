import pytest
import torch.nn

import lightning.fabric
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from tests_pytorch.helpers.runif import RunIf


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


@RunIf(min_torch="1.13")
def test_fsdp_activation_checkpointing(monkeypatch):
    with pytest.raises(ValueError, match="cannot set both `activation_checkpointing"):
        FSDPStrategy(activation_checkpointing=torch.nn.Linear, activation_checkpointing_policy=lambda *_: True)

    monkeypatch.setattr(lightning.fabric.strategies.fsdp, "_TORCH_GREATER_EQUAL_2_1", True)
    with pytest.deprecated_call(match=r"use `FSDPStrategy\(activation_checkpointing_policy"):
        FSDPStrategy(activation_checkpointing=torch.nn.Linear)
