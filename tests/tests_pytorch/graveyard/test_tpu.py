from importlib import import_module

import pytest
import torch


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
