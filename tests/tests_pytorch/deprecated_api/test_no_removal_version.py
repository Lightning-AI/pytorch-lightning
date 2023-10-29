
import lightning.fabric
import pytest
import torch.nn
from lightning.fabric.plugins.precision.bitsandbytes import _BITSANDBYTES_AVAILABLE
from lightning.fabric.plugins.precision.transformer_engine import _TRANSFORMER_ENGINE_AVAILABLE
from lightning.pytorch import Trainer
from lightning.pytorch.demos.boring_classes import BoringModel
from lightning.pytorch.plugins.precision.double import LightningDoublePrecisionModule
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


def test_double_precision_wrapper():
    with pytest.deprecated_call(match=r"The `LightningDoublePrecisionModule` is deprecated and no longer needed"):
        LightningDoublePrecisionModule(BoringModel())


def test_fsdp_mixed_precision_plugin():
    from lightning.pytorch.plugins.precision.fsdp import FSDPMixedPrecisionPlugin

    with pytest.deprecated_call(match=r"The `FSDPMixedPrecisionPlugin` is deprecated"):
        FSDPMixedPrecisionPlugin(precision="16-mixed", device="cuda")


def test_fsdp_precision_plugin():
    from lightning.pytorch.plugins.precision.fsdp import FSDPPrecisionPlugin

    with pytest.deprecated_call(match=r"The `FSDPPrecisionPlugin` is deprecated"):
        FSDPPrecisionPlugin(precision="16-mixed")


@pytest.mark.skipif(not _BITSANDBYTES_AVAILABLE, reason="bitsandbytes unavailable")
def test_bitsandbytes_precision_plugin():
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecisionPlugin

    with pytest.deprecated_call(match=r"The `BitsandbytesPrecisionPlugin` is deprecated"):
        BitsandbytesPrecisionPlugin("nf4")


@RunIf(deepspeed=True)
def test_deepspeed_precision_plugin():
    from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin

    with pytest.deprecated_call(match=r"The `DeepSpeedPrecisionPlugin` is deprecated"):
        DeepSpeedPrecisionPlugin(precision="32-true")


def test_double_precision_plugin():
    from lightning.pytorch.plugins.precision.double import DoublePrecisionPlugin

    with pytest.deprecated_call(match=r"The `DoublePrecisionPlugin` is deprecated"):
        DoublePrecisionPlugin()


def test_half_precision_plugin():
    from lightning.pytorch.plugins.precision.half import HalfPrecisionPlugin

    with pytest.deprecated_call(match=r"The `HalfPrecisionPlugin` is deprecated"):
        HalfPrecisionPlugin()


def test_mixed_precision_plugin():
    from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin

    with pytest.deprecated_call(match=r"The `MixedPrecisionPlugin` is deprecated"):
        MixedPrecisionPlugin(precision="16-mixed", device="cuda")


def test_precision_plugin():
    from lightning.pytorch.plugins.precision.precision import PrecisionPlugin

    with pytest.deprecated_call(match=r"The `PrecisionPlugin` is deprecated"):
        PrecisionPlugin()


@pytest.mark.skipif(not _TRANSFORMER_ENGINE_AVAILABLE, reason="transformer-engine unavailable")
def test_transformer_engine_precision_plugin():
    from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecisionPlugin

    with pytest.deprecated_call(match=r"The `TransformerEnginePrecisionPlugin` is deprecated"):
        TransformerEnginePrecisionPlugin()


@RunIf(tpu=True)
def test_xla_precision_plugin():
    from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin

    with pytest.deprecated_call(match=r"The `XLAPrecisionPlugin` is deprecated"):
        XLAPrecisionPlugin()
