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
def test_extracted_hpu(import_path,name):
    module = import_module(import_path)
    cls = getattr(module, name)
    with pytest.raises(NotImplementedError, match=f"{name}` class has been moved to an external package.*"):
        cls()


def test_extracted_hpu_classes_inner_import():
    from lightning.pytorch.accelerators.hpu import HPUAccelerator
    from lightning.pytorch.strategies.hpu_parallel import HPUParallelStrategy
    from lightning.pytorch.strategies.single_hpu import SingleHPUStrategy
    from lightning.pytorch.plugins.io.hpu_plugin import HPUCheckpointIO
    from lightning.pytorch.plugins.precision.hpu import HPUPrecisionPlugin

    with pytest.raises(NotImplementedError, match="HPUAccelerator` class has been moved to an external package.*"):
        HPUAccelerator()
    with pytest.raises(NotImplementedError, match="HPUParallelStrategy` class has been moved to an external package.*"):
        HPUParallelStrategy()
    with pytest.raises(NotImplementedError, match="SingleHPUStrategy` class has been moved to an external package.*"):
        SingleHPUStrategy()
    with pytest.raises(NotImplementedError, match="HPUCheckpointIO` class has been moved to an external package.*"):
        HPUCheckpointIO()
    with pytest.raises(NotImplementedError, match="HPUPrecisionPlugin` class has been moved to an external package.*"):
        HPUPrecisionPlugin()
