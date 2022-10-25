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
from lightning_lite.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning_lite.strategies.fairscale import _FAIRSCALE_AVAILABLE


def test_imports():
    try:
        import deepspeed  # noqa
    except ModuleNotFoundError:
        assert not _DEEPSPEED_AVAILABLE
    else:
        assert _DEEPSPEED_AVAILABLE

    try:
        import fairscale.nn  # noqa
    except ModuleNotFoundError:
        assert not _FAIRSCALE_AVAILABLE
    else:
        assert _FAIRSCALE_AVAILABLE
