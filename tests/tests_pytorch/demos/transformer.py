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
# limitations under the License.
import torch
from lightning.pytorch.demos import Transformer


def test_compile_transformer():
    """Smoke test to ensure the transformer model compiles without errors."""
    transformer = Transformer(vocab_size=8)
    transformer = torch.compile(transformer)
    inputs = torch.randint(0, transformer.vocab_size, size=(2, 8))
    targets = torch.randint(0, transformer.vocab_size, size=(2, 8))
    for i in range(3):
        transformer(inputs, targets).sum().backward()
