from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("import_path", "name"),
    [
        ("lightning.pytorch.accelerators", "HPUAccelerator"),
        ("lightning.pytorch.accelerators.hpu", "HPUAccelerator"),
        ("lightning.pytorch.strategies", "HPUParallelStrategy"),
        ("lightning.pytorch.strategies.hpu_parallel", "HPUParallelStrategy"),
        ("lightning.pytorch.strategies", "SingleHPUStrategy"),
        ("lightning.pytorch.strategies.single_hpu", "SingleHPUStrategy"),
        ("lightning.pytorch.plugins.io", "HPUCheckpointIO"),
        ("lightning.pytorch.plugins.io.hpu_plugin", "HPUCheckpointIO"),
        ("lightning.pytorch.plugins.precision", "HPUPrecisionPlugin"),
        ("lightning.pytorch.plugins.precision.hpu", "HPUPrecisionPlugin"),
    ],
)
def test_extracted_hpu(import_path, name):
    module = import_module(import_path)
    cls = getattr(module, name)
    with pytest.raises(NotImplementedError, match=f"{name}` class has been moved to an external package.*"):
        cls()
