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
"""Test deprecated functionality which will be removed in v1.5.0"""
import pytest

from pytorch_lightning import Trainer
from pytorch_lightning.core.decorators import auto_move_data


def test_v1_5_0_auto_move_data():
    with pytest.deprecated_call(match="deprecated in v1.3 and will be removed in v1.5.*was applied to `bar`"):

        class Foo:
            @auto_move_data
            def bar(self):
                pass


def test_v1_5_0_distributed_backend_trainer_flag():
    with pytest.deprecated_call(match="has been deprecated and will be removed in v1.5."):
        Trainer(distributed_backend="ddp_cpu")
