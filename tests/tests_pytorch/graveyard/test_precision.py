def test_precision_plugin_renamed_imports():
    # base class
    from lightning.pytorch.plugins import PrecisionPlugin as PrecisionPlugin2
    from lightning.pytorch.plugins.precision import PrecisionPlugin as PrecisionPlugin1
    from lightning.pytorch.plugins.precision.precision import Precision
    from lightning.pytorch.plugins.precision.precision_plugin import PrecisionPlugin as PrecisionPlugin0

    assert PrecisionPlugin0 is PrecisionPlugin1 is PrecisionPlugin2 is Precision

    # bitsandbytes
    from lightning.pytorch.plugins import BitsandbytesPrecisionPlugin as BnbPlugin2
    from lightning.pytorch.plugins.precision import BitsandbytesPrecisionPlugin as BnbPlugin1
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecision
    from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecisionPlugin as BnbPlugin0

    assert BnbPlugin0 is BnbPlugin1 is BnbPlugin2 is BitsandbytesPrecision

    # deepspeed
    from lightning.pytorch.plugins import DeepSpeedPrecisionPlugin as DeepSpeedPlugin2
    from lightning.pytorch.plugins.precision import DeepSpeedPrecisionPlugin as DeepSpeedPlugin1
    from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecision
    from lightning.pytorch.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin as DeepSpeedPlugin0

    assert DeepSpeedPlugin0 is DeepSpeedPlugin1 is DeepSpeedPlugin2 is DeepSpeedPrecision

    # double
    from lightning.pytorch.plugins import DoublePrecisionPlugin as DoublePlugin2
    from lightning.pytorch.plugins.precision import DoublePrecisionPlugin as DoublePlugin1
    from lightning.pytorch.plugins.precision.double import DoublePrecision
    from lightning.pytorch.plugins.precision.double import DoublePrecisionPlugin as DoublePlugin0

    assert DoublePlugin0 is DoublePlugin1 is DoublePlugin2 is DoublePrecision

    # fsdp
    from lightning.pytorch.plugins import FSDPPrecisionPlugin as FSDPPlugin2
    from lightning.pytorch.plugins.precision import FSDPPrecisionPlugin as FSDPPlugin1
    from lightning.pytorch.plugins.precision.fsdp import FSDPPrecision
    from lightning.pytorch.plugins.precision.fsdp import FSDPPrecisionPlugin as FSDPPlugin0

    assert FSDPPlugin0 is FSDPPlugin1 is FSDPPlugin2 is FSDPPrecision

    # half
    from lightning.pytorch.plugins import HalfPrecisionPlugin as HalfPlugin2
    from lightning.pytorch.plugins.precision import HalfPrecisionPlugin as HalfPlugin1
    from lightning.pytorch.plugins.precision.half import HalfPrecision
    from lightning.pytorch.plugins.precision.half import HalfPrecisionPlugin as HalfPlugin0

    assert HalfPlugin0 is HalfPlugin1 is HalfPlugin2 is HalfPrecision

    # mixed
    from lightning.pytorch.plugins import MixedPrecisionPlugin as MixedPlugin2
    from lightning.pytorch.plugins.precision import MixedPrecisionPlugin as MixedPlugin1
    from lightning.pytorch.plugins.precision.amp import MixedPrecision
    from lightning.pytorch.plugins.precision.amp import MixedPrecisionPlugin as MixedPlugin0

    assert MixedPlugin0 is MixedPlugin1 is MixedPlugin2 is MixedPrecision

    # transformer_engine
    from lightning.pytorch.plugins import TransformerEnginePrecisionPlugin as TEPlugin2
    from lightning.pytorch.plugins.precision import TransformerEnginePrecisionPlugin as TEPlugin1
    from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecision
    from lightning.pytorch.plugins.precision.transformer_engine import TransformerEnginePrecisionPlugin as TEPlugin0

    assert TEPlugin0 is TEPlugin1 is TEPlugin2 is TransformerEnginePrecision

    # xla
    from lightning.pytorch.plugins import XLAPrecisionPlugin as XLAPlugin2
    from lightning.pytorch.plugins.precision import XLAPrecisionPlugin as XLAPlugin1
    from lightning.pytorch.plugins.precision.xla import XLAPrecision
    from lightning.pytorch.plugins.precision.xla import XLAPrecisionPlugin as XLAPlugin0

    assert XLAPlugin0 is XLAPlugin1 is XLAPlugin2 is XLAPrecision
