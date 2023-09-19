from importlib import import_module

import pytest
import torch


@pytest.mark.parametrize(
    ("import_path", "name"),
    [
        ("lightning.fabric.strategies", "SingleTPUStrategy"),
        ("lightning.fabric.strategies.single_tpu", "SingleTPUStrategy"),
    ],
)
def test_graveyard_single_tpu(import_path, name):
    module = import_module(import_path)
    cls = getattr(module, name)
    device = torch.device("cpu")
    with pytest.deprecated_call(match="is deprecated"), pytest.raises(ModuleNotFoundError, match="torch_xla"):
        cls(device)


@pytest.mark.parametrize(
    ("import_path", "name"),
    [
        ("lightning.fabric.accelerators", "TPUAccelerator"),
        ("lightning.fabric.accelerators.tpu", "TPUAccelerator"),
        ("lightning.fabric.plugins", "TPUPrecision"),
        ("lightning.fabric.plugins.precision", "TPUPrecision"),
        ("lightning.fabric.plugins.precision.tpu", "TPUPrecision"),
        ("lightning.fabric.plugins", "TPUBf16Precision"),
        ("lightning.fabric.plugins.precision", "TPUBf16Precision"),
        ("lightning.fabric.plugins.precision.tpu_bf16", "TPUBf16Precision"),
        ("lightning.fabric.plugins.precision", "XLABf16Precision"),
        ("lightning.fabric.plugins.precision.xlabf16", "XLABf16Precision"),
    ],
)
def test_graveyard_no_device(import_path, name):
    module = import_module(import_path)
    cls = getattr(module, name)
    with pytest.deprecated_call(match="is deprecated"), pytest.raises(ModuleNotFoundError, match="torch_xla"):
        cls()
