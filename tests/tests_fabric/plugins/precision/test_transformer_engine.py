# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License
import sys
from unittest.mock import Mock

import lightning.fabric
import pytest
import torch
import torch.distributed
from lightning.fabric.connector import _Connector
from lightning.fabric.plugins.precision.transformer_engine import TransformerEnginePrecision


def test_transformer_engine_plugin(monkeypatch):
    module = lightning.fabric.plugins.precision.transformer_engine
    if module._TRANSFORMER_ENGINE_AVAILABLE:
        pytest.skip("Assumes transformer_engine is unavailable")
    monkeypatch.setattr(module, "_TRANSFORMER_ENGINE_AVAILABLE", lambda: True)
    transformer_engine_mock = Mock()
    monkeypatch.setitem(sys.modules, "transformer_engine", transformer_engine_mock)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", Mock())
    recipe_mock = Mock()
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", recipe_mock)

    connector = _Connector(precision="transformer-engine")
    assert isinstance(connector.precision, TransformerEnginePrecision)
    assert connector.precision.weights_dtype is torch.bfloat16
    connector = _Connector(precision="transformer-engine-float16")
    assert connector.precision.weights_dtype is torch.float16

    recipe_mock.reset_mock()
    precision = TransformerEnginePrecision(weights_dtype=torch.float32)
    connector = _Connector(plugins=precision)
    assert connector.precision is precision
    assert precision.weights_dtype == torch.float32
    recipe_mock.DelayedScaling.assert_called_once_with()

    recipe_mock.reset_mock()
    recipe = {"foo": 0, "fp8_format": "HYBRID"}
    precision = TransformerEnginePrecision(weights_dtype=torch.float16, recipe=recipe)
    connector = _Connector(plugins=precision)
    assert connector.precision is precision
    recipe_mock.DelayedScaling.assert_called_once_with(foo=0, fp8_format=recipe_mock.Format.HYBRID)
    assert isinstance(recipe["fp8_format"], str)  # not modified

    # same logic as in `test_default_dtype_is_restored`
    assert torch.get_default_dtype() is torch.float32
    with pytest.raises(RuntimeError, match="foo"), precision.module_init_context():
        assert torch.get_default_dtype() is not torch.float32
        raise RuntimeError("foo")
    assert torch.get_default_dtype() is torch.float32

    class SubModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l = torch.nn.Linear(1, 3)

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = torch.nn.Linear(16, 48)
            self.l2 = torch.nn.LayerNorm(1)
            self.l3 = SubModule()

    model = MyModule()

    precision.replace_layers = False
    precision.convert_module(model)
    assert isinstance(model.l1, torch.nn.Linear)
    assert model.l1.weight.dtype == torch.float16
    assert isinstance(model.l3.l, torch.nn.Linear)
    assert isinstance(model.l2, torch.nn.LayerNorm)

    precision.replace_layers = True
    setattr_mock = Mock()
    model.__setattr__ = setattr_mock
    with pytest.warns(match="divisible by 8 and 16"):
        precision.convert_module(model)
    mock_calls = setattr_mock.mock_calls
    assert len(mock_calls) == 2
    assert mock_calls[0][1][0] == "l1"
    assert mock_calls[1][1][0] == "l2"
    assert mock_calls[0][1][1]._extract_mock_name() == "mock.pytorch.Linear()"
    assert mock_calls[1][1][1]._extract_mock_name() == "mock.pytorch.LayerNorm()"

    precision.replace_layers = False
    with precision.module_init_context():
        model = MyModule()
    assert isinstance(model.l1, torch.nn.Linear)
    assert isinstance(model.l2, torch.nn.LayerNorm)
    assert isinstance(model.l3.l, torch.nn.Linear)

    class TELinearMock(Mock): ...

    class TELayerNormMock(Mock): ...

    transformer_engine_mock.pytorch.Linear = TELinearMock
    transformer_engine_mock.pytorch.LayerNorm = TELayerNormMock
    precision.replace_layers = True
    with precision.module_init_context():
        assert torch.get_default_dtype() == torch.float16
        model = MyModule()
    assert isinstance(model.l1, TELinearMock)
    assert isinstance(model.l2, TELayerNormMock)
    assert isinstance(model.l3.l, TELinearMock)
