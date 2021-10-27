# Copyright The PyTorch Lightning team.
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
# limitations under the License.
import operator

from pytorch_lightning.utilities import _module_available
from pytorch_lightning.utilities.imports import _compare_version


def test_module_exists():
    """Test if the some 3rd party libs are available."""
    assert _module_available("torch")
    assert _module_available("torch.nn.parallel")
    assert not _module_available("torch.nn.asdf")
    assert not _module_available("asdf")
    assert not _module_available("asdf.bla.asdf")


def test_compare_version(monkeypatch):
    from pytorch_lightning.utilities.imports import torch

    monkeypatch.setattr(torch, "__version__", "1.8.9")
    assert not _compare_version("torch", operator.ge, "1.10.0")
    assert _compare_version("torch", operator.lt, "1.10.0")

    monkeypatch.setattr(torch, "__version__", "1.10.0.dev123")
    assert _compare_version("torch", operator.ge, "1.10.0.dev123")
    assert not _compare_version("torch", operator.ge, "1.10.0.dev124")

    assert _compare_version("torch", operator.ge, "1.10.0.dev123", use_base_version=True)
    assert _compare_version("torch", operator.ge, "1.10.0.dev124", use_base_version=True)

    monkeypatch.setattr(torch, "__version__", "1.10.0a0+0aef44c")  # dev version before rc
    assert _compare_version("torch", operator.ge, "1.10.0.rc0", use_base_version=True)
    assert not _compare_version("torch", operator.ge, "1.10.0.rc0")
    assert _compare_version("torch", operator.ge, "1.10.0", use_base_version=True)
    assert not _compare_version("torch", operator.ge, "1.10.0")
