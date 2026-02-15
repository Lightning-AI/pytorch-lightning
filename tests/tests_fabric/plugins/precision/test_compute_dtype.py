import torch

from lightning.fabric.plugins.precision.amp import MixedPrecision
from lightning.fabric.plugins.precision.bitsandbytes import BitsandbytesPrecision
from lightning.fabric.plugins.precision.deepspeed import DeepSpeedPrecision
from lightning.fabric.plugins.precision.double import DoublePrecision
from lightning.fabric.plugins.precision.fsdp import FSDPPrecision
from lightning.fabric.plugins.precision.half import HalfPrecision
from lightning.fabric.plugins.precision.precision import Precision
from lightning.fabric.plugins.precision.transformer_engine import TransformerEnginePrecision
from lightning.fabric.plugins.precision.xla import XLAPrecision


def test_precision_compute_dtype_default():
    assert Precision().compute_dtype() is torch.float32


def test_precision_compute_dtype_half_and_mixed():
    assert HalfPrecision("16-true").compute_dtype() is torch.float16
    assert HalfPrecision("bf16-true").compute_dtype() is torch.bfloat16
    assert MixedPrecision("16-mixed", device="cpu").compute_dtype() is torch.float16
    assert MixedPrecision("bf16-mixed", device="cpu").compute_dtype() is torch.bfloat16


def test_precision_compute_dtype_double():
    assert DoublePrecision().compute_dtype() is torch.double


def test_precision_compute_dtype_bitsandbytes_without_init():
    plugin = object.__new__(BitsandbytesPrecision)
    plugin.dtype = torch.bfloat16
    assert BitsandbytesPrecision.compute_dtype(plugin) is torch.bfloat16


def test_precision_compute_dtype_fsdp_and_deepspeed():
    assert FSDPPrecision("16-mixed").compute_dtype() is torch.float16
    assert FSDPPrecision("bf16-mixed").compute_dtype() is torch.bfloat16
    assert DeepSpeedPrecision("32-true").compute_dtype() is torch.float32
    assert DeepSpeedPrecision("16-mixed").compute_dtype() is torch.float16


def test_precision_compute_dtype_xla_without_init():
    plugin = object.__new__(XLAPrecision)
    plugin._desired_dtype = torch.bfloat16
    assert XLAPrecision.compute_dtype(plugin) is torch.bfloat16


def test_precision_compute_dtype_transformer_engine_without_init():
    plugin = object.__new__(TransformerEnginePrecision)
    assert TransformerEnginePrecision.compute_dtype(plugin) is torch.int8
