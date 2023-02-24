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
from lightning.fabric.utilities.types import _Stateful


def test_stateful_protocol():
    class StatefulClass:
        def state_dict(self):
            pass

        def load_state_dict(self, state_dict):
            pass

    assert isinstance(StatefulClass(), _Stateful)

    class NotStatefulClass:
        def state_dict(self):
            pass

    assert not isinstance(NotStatefulClass(), _Stateful)

    class NotStateful2Class:
        def load_state_dict(self, state_dict):
            pass

    assert not isinstance(NotStateful2Class(), _Stateful)
