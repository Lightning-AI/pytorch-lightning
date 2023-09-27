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
from lightning.fabric.plugins import (
    BitsandbytesPrecision,
)


def test_bitsandbytes_plugin(monkeypatch):
    module = lightning.fabric.plugins.precision.bitsandbytes
    if module._BITSANDBYTES_AVAILABLE:
        pytest.skip("Assumes bitsandbytes is unavailable")
    monkeypatch.setattr(module, "_BITSANDBYTES_AVAILABLE", lambda: True)
    bitsandbytes_mock = Mock()
    monkeypatch.setitem(sys.modules, "bitsandbytes", bitsandbytes_mock)

    class ModuleMock(torch.nn.Module):
        def __init__(self, *_, **__):
            super().__init__()

    class NF4LinearMock(ModuleMock):
        ...

    module._FP4Linear = ModuleMock
    module._NF4Linear = NF4LinearMock
    module._Int8LinearInference = ModuleMock
    module._FP4DQLinear = ModuleMock
    module._NF4DQLinear = ModuleMock
    module._Linear4bit = ModuleMock
    module._Linear8bitLt = ModuleMock

    precision = BitsandbytesPrecision("nf4", dtype=torch.float16)
    connector = _Connector(plugins=precision)
    assert connector.precision is precision
    assert precision.dtype == torch.float16

    # same logic as in `test_default_dtype_is_restored`
    assert torch.get_default_dtype() is torch.float32
    with pytest.raises(RuntimeError, match="foo"), precision.init_context():
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
            self.l2 = SubModule()

    with precision.init_context():
        assert torch.get_default_dtype() == torch.float16
        model = MyModule()
    assert isinstance(model.l1, NF4LinearMock)
    assert isinstance(model.l2.l, NF4LinearMock)
    model = precision.convert_module(model)
    assert model.l1.compute_dtype is precision.dtype
    assert model.l2.l.compute_dtype is precision.dtype
