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
import importlib


def _module_available(module_path: str) -> bool:
    """Testing if given module is avalaible in your env

    >>> _module_available('os')
    True
    >>> _module_available('bla.bla')
    False
    """
    # todo: find a better way than try / except
    try:
        mods = module_path.split('.')
        assert mods, 'nothing given to test'
        # it has to be tested as per partets
        for i in range(len(mods)):
            module_path = '.'.join(mods[:i + 1])
            if importlib.util.find_spec(module_path) is None:
                return False
        return True
    except AttributeError:
        return False
