from importlib import import_module

import pytest
import torch


@pytest.mark.parametrize(
    ("import_path", "name"),
    [
        ("lightning_pytorch.strategies", "SingleTPUStrategy"),
        ("lightning_pytorch.strategies.single_tpu", "SingleTPUStrategy"),
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
        ("lightning_pytorch.accelerators", "TPUAccelerator"),
        ("lightning_pytorch.accelerators.tpu", "TPUAccelerator"),
        ("lightning_pytorch.plugins", "TPUPrecisionPlugin"),
        ("lightning_pytorch.plugins.precision", "TPUPrecisionPlugin"),
        ("lightning_pytorch.plugins.precision.tpu", "TPUPrecisionPlugin"),
        ("lightning_pytorch.plugins", "TPUBf16PrecisionPlugin"),
        ("lightning_pytorch.plugins.precision", "TPUBf16PrecisionPlugin"),
        ("lightning_pytorch.plugins.precision.tpu_bf16", "TPUBf16PrecisionPlugin"),
        ("lightning_pytorch.plugins.precision", "XLABf16PrecisionPlugin"),
        ("lightning_pytorch.plugins.precision.xlabf16", "XLABf16PrecisionPlugin"),
    ],
)
def test_graveyard_no_device(import_path, name):
    module = import_module(import_path)
    cls = getattr(module, name)
    with pytest.deprecated_call(match="is deprecated"), pytest.raises(ModuleNotFoundError, match="torch_xla"):
        cls()
