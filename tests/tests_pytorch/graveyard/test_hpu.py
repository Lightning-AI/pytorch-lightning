from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("import_path", "name"),
    [
        ("lightning_pytorch.accelerators", "HPUAccelerator"),
        ("lightning_pytorch.accelerators.hpu", "HPUAccelerator"),
        ("lightning_pytorch.strategies", "HPUParallelStrategy"),
        ("lightning_pytorch.strategies.hpu_parallel", "HPUParallelStrategy"),
        ("lightning_pytorch.strategies", "SingleHPUStrategy"),
        ("lightning_pytorch.strategies.single_hpu", "SingleHPUStrategy"),
        ("lightning_pytorch.plugins.io", "HPUCheckpointIO"),
        ("lightning_pytorch.plugins.io.hpu_plugin", "HPUCheckpointIO"),
        ("lightning_pytorch.plugins.precision", "HPUPrecisionPlugin"),
        ("lightning_pytorch.plugins.precision.hpu", "HPUPrecisionPlugin"),
    ],
)
def test_extracted_hpu(import_path, name):
    module = import_module(import_path)
    cls = getattr(module, name)
    with pytest.raises(NotImplementedError, match=f"{name}` class has been moved to an external package.*"):
        cls()
