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
from lightning.fabric.plugins.precision.bitsandbytes import _BITSANDBYTES_AVAILABLE
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.plugins.precision.bitsandbytes import BitsandbytesPrecision


@pytest.mark.skipif(_BITSANDBYTES_AVAILABLE, reason="bitsandbytes needs to be unavailable")
def test_bitsandbytes_plugin(monkeypatch):
    module = lightning.fabric.plugins.precision.bitsandbytes
    monkeypatch.setattr(module, "_BITSANDBYTES_AVAILABLE", lambda: True)
    bitsandbytes_mock = Mock()
    monkeypatch.setitem(sys.modules, "bitsandbytes", bitsandbytes_mock)

    class ModuleMock(torch.nn.Linear):
        def __init__(self, in_features, out_features, bias=True, *_, **__):
            super().__init__(in_features, out_features, bias)

    bitsandbytes_mock.nn.Linear8bitLt = ModuleMock
    bitsandbytes_mock.nn.Linear4bit = ModuleMock
    bitsandbytes_mock.nn.Params4bit = object

    precision = BitsandbytesPrecision("nf4", dtype=torch.float16)
    trainer = Trainer(barebones=True, plugins=precision)

    _NF4Linear = vars(module)["_NF4Linear"]
    quantize_mock = lambda self, p, w, d: p
    _NF4Linear.quantize = quantize_mock

    class MyModel(LightningModule):
        def configure_model(self):
            self.l = torch.nn.Linear(1, 3)

        def test_step(self, *_): ...

    model = MyModel()
    trainer.test(model, [0])
    assert isinstance(model.l, _NF4Linear)
