import pytest
import torch
from lightning.fabric.plugins.precision.utils import _ClassReplacementContextManager, _DtypeContextManager


def test_dtype_context_manager():
    # regular issue
    assert torch.get_default_dtype() is torch.float32
    with _DtypeContextManager(torch.float16):
        assert torch.get_default_dtype() is torch.float16

    # exception
    assert torch.get_default_dtype() is torch.float32
    with pytest.raises(RuntimeError, match="foo"), _DtypeContextManager(torch.float16):
        assert torch.get_default_dtype() is torch.float16
        raise RuntimeError("foo")
    assert torch.get_default_dtype() is torch.float32


def test_class_replacement_context_manager():
    original_linear = torch.nn.Linear
    original_layernorm = torch.nn.LayerNorm

    class MyLinear:
        def __init__(self, *_, **__):
            pass

    class MyLayerNorm:
        def __init__(self, *_, **__):
            pass

    context_manager = _ClassReplacementContextManager({"torch.nn.Linear": MyLinear, "torch.nn.LayerNorm": MyLayerNorm})
    assert context_manager._originals == {"torch.nn.Linear": original_linear, "torch.nn.LayerNorm": original_layernorm}
    assert context_manager._modules == {"torch.nn.Linear": torch.nn, "torch.nn.LayerNorm": torch.nn}

    with context_manager:
        linear = torch.nn.Linear(100, 100)
        layernorm = torch.nn.LayerNorm(1)
    assert isinstance(linear, MyLinear)
    assert isinstance(layernorm, MyLayerNorm)
    assert not hasattr(linear, "forward")

    linear = torch.nn.Linear(100, 100)
    layernorm = torch.nn.LayerNorm(1)
    assert isinstance(linear, original_linear)
    assert isinstance(layernorm, original_layernorm)
    assert hasattr(linear, "forward")
