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

import torch

from lightning_lite.plugins.precision.double import DoublePrecision


def test_double_precision_forward_context():
    precision = DoublePrecision()
    assert torch.get_default_dtype() == torch.float32
    with precision.forward_context():
        assert torch.get_default_dtype() == torch.float64
    assert torch.get_default_dtype() == torch.float32
