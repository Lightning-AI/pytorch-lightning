import sys

import lightning.pytorch as pl
from lightning.pytorch.plugins.precision import (
    BitsandbytesPrecision,
    DeepSpeedPrecision,
    DoublePrecision,
    FSDPPrecision,
    HalfPrecision,
    MixedPrecision,
    Precision,
    TransformerEnginePrecision,
    XLAPrecision,
)


def _patch_sys_modules() -> None:
    sys.modules["lightning.pytorch.plugins.precision.precision_plugin"] = sys.modules[
        "lightning.pytorch.plugins.precision.precision"
    ]


def _patch_classes() -> None:
    # precision file
    setattr(pl.plugins.precision.precision, "PrecisionPlugin", Precision)
    setattr(pl.plugins.precision.bitsandbytes, "BitsandbytesPrecisionPlugin", BitsandbytesPrecision)
    setattr(pl.plugins.precision.deepspeed, "DeepSpeedPrecisionPlugin", DeepSpeedPrecision)
    setattr(pl.plugins.precision.double, "DoublePrecisionPlugin", DoublePrecision)
    setattr(pl.plugins.precision.fsdp, "FSDPPrecisionPlugin", FSDPPrecision)
    setattr(pl.plugins.precision.half, "HalfPrecisionPlugin", HalfPrecision)
    setattr(pl.plugins.precision.amp, "MixedPrecisionPlugin", MixedPrecision)
    setattr(pl.plugins.precision.transformer_engine, "TransformerEnginePrecisionPlugin", TransformerEnginePrecision)
    setattr(pl.plugins.precision.xla, "XLAPrecisionPlugin", XLAPrecision)
    # precision plugin module
    setattr(pl.plugins.precision, "PrecisionPlugin", Precision)
    setattr(pl.plugins.precision, "BitsandbytesPrecisionPlugin", BitsandbytesPrecision)
    setattr(pl.plugins.precision, "DeepSpeedPrecisionPlugin", DeepSpeedPrecision)
    setattr(pl.plugins.precision, "DoublePrecisionPlugin", DoublePrecision)
    setattr(pl.plugins.precision, "FSDPPrecisionPlugin", FSDPPrecision)
    setattr(pl.plugins.precision, "HalfPrecisionPlugin", HalfPrecision)
    setattr(pl.plugins.precision, "MixedPrecisionPlugin", MixedPrecision)
    setattr(pl.plugins.precision, "TransformerEnginePrecisionPlugin", TransformerEnginePrecision)
    setattr(pl.plugins.precision, "XLAPrecisionPlugin", XLAPrecision)
    # plugins top module
    setattr(pl.plugins, "PrecisionPlugin", Precision)
    setattr(pl.plugins, "BitsandbytesPrecisionPlugin", BitsandbytesPrecision)
    setattr(pl.plugins, "DeepSpeedPrecisionPlugin", DeepSpeedPrecision)
    setattr(pl.plugins, "DoublePrecisionPlugin", DoublePrecision)
    setattr(pl.plugins, "FSDPPrecisionPlugin", FSDPPrecision)
    setattr(pl.plugins, "HalfPrecisionPlugin", HalfPrecision)
    setattr(pl.plugins, "MixedPrecisionPlugin", MixedPrecision)
    setattr(pl.plugins, "TransformerEnginePrecisionPlugin", TransformerEnginePrecision)
    setattr(pl.plugins, "XLAPrecisionPlugin", XLAPrecision)


_patch_sys_modules()
_patch_classes()
