import sys
from typing import TYPE_CHECKING, Any, Literal, Optional

import lightning.pytorch as pl
from lightning.fabric.utilities.rank_zero import rank_zero_deprecation
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

if TYPE_CHECKING:
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler


def _patch_sys_modules() -> None:
    sys.modules["lightning.pytorch.plugins.precision.precision_plugin"] = sys.modules[
        "lightning.pytorch.plugins.precision.precision"
    ]


class FSDPMixedPrecisionPlugin(FSDPPrecision):
    """AMP for Fully Sharded Data Parallel (FSDP) Training.

    .. deprecated:: Use :class:`FSDPPrecision` instead.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(
        self, precision: Literal["16-mixed", "bf16-mixed"], device: str, scaler: Optional["ShardedGradScaler"] = None
    ) -> None:
        rank_zero_deprecation(
            f"The `{type(self).__name__}` is deprecated."
            " Use `lightning.pytorch.plugins.precision.FSDPPrecision` instead."
        )
        super().__init__(precision=precision, scaler=scaler)


def _create_class(deprecated_name: str, new_class: type) -> type:
    def init(self: type, *args: Any, **kwargs: Any) -> None:
        rank_zero_deprecation(
            f"The `{deprecated_name}` is deprecated."
            f" Use `lightning.pytorch.plugins.precision.{new_class.__name__}` instead."
        )
        new_class.__init__(self, *args, **kwargs)  # type: ignore[misc]

    return type(deprecated_name, (new_class,), {"__init__": init})


def _patch_classes() -> None:
    classes_map = (
        # module name, old name, new class
        ("bitsandbytes", "BitsandbytesPrecisionPlugin", BitsandbytesPrecision),
        ("deepspeed", "DeepSpeedPrecisionPlugin", DeepSpeedPrecision),
        ("double", "DoublePrecisionPlugin", DoublePrecision),
        ("fsdp", "FSDPPrecisionPlugin", FSDPPrecision),
        ("fsdp", "FSDPMixedPrecisionPlugin", FSDPPrecision),
        ("half", "HalfPrecisionPlugin", HalfPrecision),
        ("amp", "MixedPrecisionPlugin", MixedPrecision),
        ("precision", "PrecisionPlugin", Precision),
        ("transformer_engine", "TransformerEnginePrecisionPlugin", TransformerEnginePrecision),
        ("xla", "XLAPrecisionPlugin", XLAPrecision),
    )

    for module_name, deprecated_name, new_class in classes_map:
        deprecated_class = _create_class(deprecated_name, new_class)
        setattr(getattr(pl.plugins.precision, module_name), deprecated_name, deprecated_class)
        setattr(pl.plugins.precision, deprecated_name, deprecated_class)
        setattr(pl.plugins, deprecated_name, deprecated_class)

    # special treatment for `FSDPMixedPrecisionPlugin` because it has a different signature
    setattr(pl.plugins.precision.fsdp, "FSDPMixedPrecisionPlugin", FSDPMixedPrecisionPlugin)
    setattr(pl.plugins.precision, "FSDPMixedPrecisionPlugin", FSDPMixedPrecisionPlugin)
    setattr(pl.plugins, "FSDPMixedPrecisionPlugin", FSDPMixedPrecisionPlugin)


_patch_sys_modules()
_patch_classes()
