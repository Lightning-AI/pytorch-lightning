

def test_precision_plugin_renamed_imports():
    # base class
    from lightning.pytorch.plugins.precision.precision import Precision
    from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin as PrecisionPlugin0
    from lightning.pytorch.plugins.precision import PrecisionPlugin as PrecisionPlugin1
    from lightning.pytorch.plugins import PrecisionPlugin as PrecisionPlugin2

    assert Precision is PrecisionPlugin0 is PrecisionPlugin1 is PrecisionPlugin2
    assert isinstance(PrecisionPlugin0(), Precision)

    # bitsandbytes
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecision
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecisionPlugin as BnbPlugin0
    from lightning.pytorch.plugins.precision import BitsandbytesPrecisionPlugin as BnbPlugin1
    from lightning.pytorch.plugins import BitsandbytesPrecisionPlugin as BnbPlugin2

    assert BitsandbytesPrecision is BnbPlugin0 is BnbPlugin1 is BnbPlugin2

    # deepspeed
    from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecision
    from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin as DeepSpeedPlugin0
    from lightning.pytorch.plugins.precision import DeepSpeedPrecisionPlugin as DeepSpeedPlugin1
    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin as DeepSpeedPlugin2

    assert DeepSpeedPrecision is DeepSpeedPlugin0 is DeepSpeedPlugin1 is DeepSpeedPlugin2

    # double
    from lightning.pytorch.plugins.precision.double import DoublePrecision
    from lightning.pytorch.plugins.precision.double import DoublePrecisionPlugin as DoublePlugin0
    from lightning.pytorch.plugins.precision import DoublePrecisionPlugin as DoublePlugin1
    from lightning.pytorch.plugins import DoublePrecisionPlugin as DoublePlugin2

    assert DoublePrecision is DoublePlugin0 is DoublePlugin1 is DoublePlugin2

    # fsdp
    from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
    from lightning.pytorch.plugins.precision.fsdp import FSDPPrecisionPlugin as FSDPPlugin0
    from lightning.pytorch.plugins.precision import FSDPPrecisionPlugin as FSDPPlugin1
    from lightning.pytorch.plugins import FSDPPrecisionPlugin as FSDPPlugin2

    assert FSDPPrecision is FSDPPlugin0 is FSDPPlugin1 is FSDPPlugin2

    # half
    from lightning.pytorch.plugins.precision.half import HalfPrecision
    from lightning.pytorch.plugins.precision.half import HalfPrecisionPlugin as HalfPlugin0
    from lightning.pytorch.plugins.precision import HalfPrecisionPlugin as HalfPlugin1
    from lightning.pytorch.plugins import HalfPrecisionPlugin as HalfPlugin2

    assert HalfPrecision is HalfPlugin0 is HalfPlugin1 is HalfPlugin2

    # mixed
    from lightning.pytorch.plugins.precision.amp import MixedPrecision
    from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin as MixedPlugin0
    from lightning.pytorch.plugins.precision import MixedPrecisionPlugin as MixedPlugin1
    from lightning.pytorch.plugins import MixedPrecisionPlugin as MixedPlugin2

    assert MixedPrecision is MixedPlugin0 is MixedPlugin1 is MixedPlugin2

    # transformer_engine
    from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecision
    from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecisionPlugin as TEPlugin0
    from lightning.pytorch.plugins.precision import TransformerEnginePrecisionPlugin as TEPlugin1
    from lightning.pytorch.plugins import TransformerEnginePrecisionPlugin as TEPlugin2

    assert TransformerEnginePrecision is TEPlugin0 is TEPlugin1 is TEPlugin2

    # xla
    from lightning.pytorch.plugins.precision.xla import XLAPrecision
    from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin as XLAPlugin0
    from lightning.pytorch.plugins.precision import XLAPrecisionPlugin as XLAPlugin1
    from lightning.pytorch.plugins import XLAPrecisionPlugin as XLAPlugin2

    assert XLAPrecision is XLAPlugin0 is XLAPlugin1 is XLAPlugin2
