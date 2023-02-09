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
import inspect

from lightning.pytorch import Trainer
from lightning.pytorch.utilities.argparse import get_init_arguments_and_types


def test_get_init_arguments_and_types():
    """Asserts a correctness of the `get_init_arguments_and_types` Trainer classmethod."""
    args = get_init_arguments_and_types(Trainer)
    parameters = inspect.signature(Trainer).parameters
    assert len(parameters) == len(args)
    for arg in args:
        assert parameters[arg[0]].default == arg[2]
