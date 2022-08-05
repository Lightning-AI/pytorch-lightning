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
import os
from unittest.mock import Mock

from pytorch_lightning.plugins import TPUBf16PrecisionPlugin


def test_teardown():
    plugin = TPUBf16PrecisionPlugin()
    plugin.connect(Mock(), Mock(), Mock())
    assert os.environ.get("XLA_USE_BF16") == "1"
    plugin.teardown()
    assert "XLA_USE_BF16" not in os.environ
