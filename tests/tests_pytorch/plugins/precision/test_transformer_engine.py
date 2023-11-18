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

import pytest
import torch
from lightning.pytorch.plugins import TransformerEnginePrecision
from lightning.pytorch.trainer.connectors.accelerator_connector import _AcceleratorConnector


def test_transformer_engine_precision_plugin(monkeypatch):
    import lightning.fabric  # avoid breakage with standalone package

    module = lightning.fabric.plugins.precision.transformer_engine
    if module._TRANSFORMER_ENGINE_AVAILABLE:
        pytest.skip("Assumes transformer_engine is unavailable")
    monkeypatch.setattr(module, "_TRANSFORMER_ENGINE_AVAILABLE", lambda: True)
    monkeypatch.setitem(sys.modules, "transformer_engine", Mock())
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", Mock())

    connector = _AcceleratorConnector(precision="transformer-engine")
    assert isinstance(connector.precision_plugin, TransformerEnginePrecision)
    assert connector.precision_plugin.dtype is torch.bfloat16
    connector = _AcceleratorConnector(precision="transformer-engine-float16")
    assert connector.precision_plugin.dtype is torch.float16

    precision = TransformerEnginePrecision()
    connector = _AcceleratorConnector(plugins=precision)
    assert connector.precision_plugin is precision
