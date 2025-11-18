import os
from importlib import import_module

import pytest
import torch


# mimics `lightning_utilites.RequirementCache`
class MockXLAAvailable:
    def __init__(self, available: bool, pkg_name: str = "torch_xla"):
        self.available = available
        self.pkg_name = pkg_name

    def __bool__(self):
        return self.available

    def __str__(self):
        if self.available:
            return f"Requirement '{self.pkg_name}' met"
        return f"Module not found: {self.pkg_name!r}. HINT: Try running `pip install -U {self.pkg_name}`"


@pytest.mark.parametrize(
    ("import_path", "name"),
    [
        ("lightning.pytorch.strategies", "SingleTPUStrategy"),
        ("lightning.pytorch.strategies.single_tpu", "SingleTPUStrategy"),
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
        ("lightning.pytorch.accelerators", "TPUAccelerator"),
        ("lightning.pytorch.accelerators.tpu", "TPUAccelerator"),
        ("lightning.pytorch.plugins", "TPUPrecisionPlugin"),
        ("lightning.pytorch.plugins.precision", "TPUPrecisionPlugin"),
        ("lightning.pytorch.plugins.precision.tpu", "TPUPrecisionPlugin"),
        ("lightning.pytorch.plugins", "TPUBf16PrecisionPlugin"),
        ("lightning.pytorch.plugins.precision", "TPUBf16PrecisionPlugin"),
        ("lightning.pytorch.plugins.precision.tpu_bf16", "TPUBf16PrecisionPlugin"),
        ("lightning.pytorch.plugins.precision", "XLABf16PrecisionPlugin"),
        ("lightning.pytorch.plugins.precision.xlabf16", "XLABf16PrecisionPlugin"),
    ],
)
def test_graveyard_no_device(import_path, name, monkeypatch):
    monkeypatch.setattr("pytorch_lightning_enterprise.accelerators.xla._XLA_AVAILABLE", MockXLAAvailable(False))
    monkeypatch.setattr("pytorch_lightning_enterprise.plugins.precision.xla._XLA_AVAILABLE", MockXLAAvailable(False))

    module = import_module(import_path)
    cls = getattr(module, name)
    with pytest.deprecated_call(match="is deprecated"), pytest.raises(ModuleNotFoundError, match="torch_xla"):
        cls()

    # teardown
    # ideally, we should call the plugin's teardown method, but since the class
    # instantiation itself fails, we directly manipulate the env vars here
    os.environ.pop("XLA_USE_BF16", None)
    os.environ.pop("XLA_USE_F16", None)
