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
from contextlib import nullcontext
from unittest.mock import ANY, Mock

import pytest
import torch

import lightning.fabric
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.plugins import TransformerEnginePrecision
from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector


def test_transformer_engine_precision_plugin(monkeypatch):
    module = lightning.fabric.plugins.precision.transformer_engine
    if module._TRANSFORMER_ENGINE_AVAILABLE:
        pytest.skip("Assumes transformer_engine is unavailable")
    monkeypatch.setattr(module, "_TRANSFORMER_ENGINE_AVAILABLE", lambda: True)
    monkeypatch.setitem(sys.modules, "transformer_engine", Mock())
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", Mock())

    connector = _AcceleratorConnector(precision="transformer-engine")
    assert isinstance(connector.precision_plugin, TransformerEnginePrecision)
    assert connector.precision_plugin.weights_dtype is torch.bfloat16
    connector = _AcceleratorConnector(precision="transformer-engine-float16")
    assert connector.precision_plugin.weights_dtype is torch.float16

    precision = TransformerEnginePrecision(weights_dtype=torch.float32)
    connector = _AcceleratorConnector(plugins=precision)
    assert connector.precision_plugin is precision


def test_configure_model(monkeypatch):
    module = lightning.fabric.plugins.precision.transformer_engine
    if module._TRANSFORMER_ENGINE_AVAILABLE:
        pytest.skip("Assumes transformer_engine is unavailable")
    monkeypatch.setattr(module, "_TRANSFORMER_ENGINE_AVAILABLE", lambda: True)
    te_mock = Mock()
    te_mock.pytorch.fp8_autocast.return_value = nullcontext()

    class LinearMock(torch.nn.Linear):
        def __init__(self, in_features, out_features, bias=True, *_, **__):
            super().__init__(in_features, out_features, bias)

    class LayerNormMock(torch.nn.LayerNorm):
        def __init__(self, normalized_shape, eps=1e-5, *_, **__):
            super().__init__(normalized_shape, eps)

    te_mock.pytorch.Linear = LinearMock
    te_mock.pytorch.LayerNorm = LayerNormMock
    monkeypatch.setitem(sys.modules, "transformer_engine", te_mock)
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", te_mock)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", te_mock)

    class LinearModel(LightningModule):
        def configure_model(self):
            self.l = torch.nn.Linear(8, 16)
            assert self.l.weight.dtype == torch.float16

        def test_step(self, *_): ...

    class LayerNormModel(LightningModule):
        def configure_model(self):
            self.l = torch.nn.LayerNorm(8, elementwise_affine=True, bias=True)
            assert self.l.weight.dtype == torch.float16

        def test_step(self, *_): ...

    class LayerNormNoAffineModel(LightningModule):
        def configure_model(self):
            self.l = torch.nn.LayerNorm(8, elementwise_affine=False)
            assert self.l.weight is None

        def test_step(self, *_): ...

    class LayerNormNoBiasModel(LightningModule):
        def configure_model(self):
            self.l = torch.nn.LayerNorm(8, elementwise_affine=True, bias=False)
            assert self.l.weight.dtype == torch.float16
            assert self.l.bias is None

        def test_step(self, *_): ...

    trainer = Trainer(barebones=True, precision="transformer-engine-float16")

    linear_model = LinearModel()
    trainer.test(linear_model, [0])
    te_mock.pytorch.fp8_autocast.assert_called_once_with(enabled=True, fp8_recipe=ANY)
    assert isinstance(linear_model.l, LinearMock)
    assert linear_model.l.weight.dtype == torch.float16

    layernorm_model = LayerNormModel()
    trainer.test(layernorm_model, [0])
    te_mock.pytorch.fp8_autocast.assert_called_with(enabled=True, fp8_recipe=ANY)
    assert isinstance(layernorm_model.l, LayerNormMock)
    assert layernorm_model.l.weight.dtype == torch.float16

    # LayerNorm should not be replaced when it has no weight and bias
    layernorm_no_affine_model = LayerNormNoAffineModel()
    trainer.test(layernorm_no_affine_model, [0])
    te_mock.pytorch.fp8_autocast.assert_called_with(enabled=True, fp8_recipe=ANY)
    assert not isinstance(layernorm_no_affine_model.l, LayerNormMock)
    assert layernorm_no_affine_model.l.weight is None

    # LayerNorm should not be replaced when it has no bias, even if it has weight
    layernorm_no_bias_model = LayerNormNoBiasModel()
    trainer.test(layernorm_no_bias_model, [0])
    te_mock.pytorch.fp8_autocast.assert_called_with(enabled=True, fp8_recipe=ANY)
    assert not isinstance(layernorm_no_bias_model.l, LayerNormMock)
    assert layernorm_no_bias_model.l.weight.dtype == torch.float16
    assert layernorm_no_bias_model.l.bias is None