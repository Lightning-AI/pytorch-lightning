import pytest


def test_precision_plugin_renamed_imports():
    # base class
    from lightning.pytorch.plugins import PrecisionPlugin as PrecisionPlugin2
    from lightning.pytorch.plugins.precision import PrecisionPlugin as PrecisionPlugin1
    from lightning.pytorch.plugins.precision.precision import Precision
    from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin as PrecisionPlugin0

    assert issubclass(PrecisionPlugin0, Precision)
    assert issubclass(PrecisionPlugin1, Precision)
    assert issubclass(PrecisionPlugin2, Precision)

    for plugin_cls in [PrecisionPlugin0, PrecisionPlugin1, PrecisionPlugin2]:
        with pytest.warns(DeprecationWarning, match="The `PrecisionPlugin` is deprecated"):
            plugin_cls()

    # bitsandbytes
    from lightning.pytorch.plugins import BitsandbytesPrecisionPlugin as BnbPlugin2
    from lightning.pytorch.plugins.precision import BitsandbytesPrecisionPlugin as BnbPlugin1
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecision
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecisionPlugin as BnbPlugin0

    assert issubclass(BnbPlugin0, BitsandbytesPrecision)
    assert issubclass(BnbPlugin1, BitsandbytesPrecision)
    assert issubclass(BnbPlugin2, BitsandbytesPrecision)

    # deepspeed
    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin as DeepSpeedPlugin2
    from lightning.pytorch.plugins.precision import DeepSpeedPrecisionPlugin as DeepSpeedPlugin1
    from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecision
    from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin as DeepSpeedPlugin0

    assert issubclass(DeepSpeedPlugin0, DeepSpeedPrecision)
    assert issubclass(DeepSpeedPlugin1, DeepSpeedPrecision)
    assert issubclass(DeepSpeedPlugin2, DeepSpeedPrecision)

    # double
    from lightning.pytorch.plugins import DoublePrecisionPlugin as DoublePlugin2
    from lightning.pytorch.plugins.precision import DoublePrecisionPlugin as DoublePlugin1
    from lightning.pytorch.plugins.precision.double import DoublePrecision
    from lightning.pytorch.plugins.precision.double import DoublePrecisionPlugin as DoublePlugin0

    assert issubclass(DoublePlugin0, DoublePrecision)
    assert issubclass(DoublePlugin1, DoublePrecision)
    assert issubclass(DoublePlugin2, DoublePrecision)

    for plugin_cls in [DoublePlugin0, DoublePlugin1, DoublePlugin2]:
        with pytest.warns(DeprecationWarning, match="The `DoublePrecisionPlugin` is deprecated"):
            plugin_cls()

    # fsdp
    from lightning.pytorch.plugins import FSDPPrecisionPlugin as FSDPPlugin2
    from lightning.pytorch.plugins.precision import FSDPPrecisionPlugin as FSDPPlugin1
    from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
    from lightning.pytorch.plugins.precision.fsdp import FSDPPrecisionPlugin as FSDPPlugin0

    assert issubclass(FSDPPlugin0, FSDPPrecision)
    assert issubclass(FSDPPlugin1, FSDPPrecision)
    assert issubclass(FSDPPlugin2, FSDPPrecision)

    for plugin_cls in [FSDPPlugin0, FSDPPlugin1, FSDPPlugin2]:
        with pytest.warns(DeprecationWarning, match="The `FSDPPrecisionPlugin` is deprecated"):
            plugin_cls(precision="16-mixed")

    # half
    from lightning.pytorch.plugins import HalfPrecisionPlugin as HalfPlugin2
    from lightning.pytorch.plugins.precision import HalfPrecisionPlugin as HalfPlugin1
    from lightning.pytorch.plugins.precision.half import HalfPrecision
    from lightning.pytorch.plugins.precision.half import HalfPrecisionPlugin as HalfPlugin0

    assert issubclass(HalfPlugin0, HalfPrecision)
    assert issubclass(HalfPlugin1, HalfPrecision)
    assert issubclass(HalfPlugin2, HalfPrecision)

    for plugin_cls in [HalfPlugin0, HalfPlugin1, HalfPlugin2]:
        with pytest.warns(DeprecationWarning, match="The `HalfPrecisionPlugin` is deprecated"):
            plugin_cls()

    # mixed
    from lightning.pytorch.plugins import MixedPrecisionPlugin as MixedPlugin2
    from lightning.pytorch.plugins.precision import MixedPrecisionPlugin as MixedPlugin1
    from lightning.pytorch.plugins.precision.amp import MixedPrecision
    from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin as MixedPlugin0

    assert issubclass(MixedPlugin0, MixedPrecision)
    assert issubclass(MixedPlugin1, MixedPrecision)
    assert issubclass(MixedPlugin2, MixedPrecision)

    for plugin_cls in [MixedPlugin0, MixedPlugin1, MixedPlugin2]:
        with pytest.warns(DeprecationWarning, match="The `MixedPrecisionPlugin` is deprecated"):
            plugin_cls(precision="bf16-mixed", device="cuda:0")

    # transformer_engine
    from lightning.pytorch.plugins import TransformerEnginePrecisionPlugin as TEPlugin2
    from lightning.pytorch.plugins.precision import TransformerEnginePrecisionPlugin as TEPlugin1
    from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecision
    from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecisionPlugin as TEPlugin0

    assert issubclass(TEPlugin0, TransformerEnginePrecision)
    assert issubclass(TEPlugin1, TransformerEnginePrecision)
    assert issubclass(TEPlugin2, TransformerEnginePrecision)

    # xla
    from lightning.pytorch.plugins import XLAPrecisionPlugin as XLAPlugin2
    from lightning.pytorch.plugins.precision import XLAPrecisionPlugin as XLAPlugin1
    from lightning.pytorch.plugins.precision.xla import XLAPrecision
    from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin as XLAPlugin0

    assert issubclass(XLAPlugin0, XLAPrecision)
    assert issubclass(XLAPlugin1, XLAPrecision)
    assert issubclass(XLAPlugin2, XLAPrecision)
