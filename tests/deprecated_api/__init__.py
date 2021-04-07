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
"""Test deprecated functionality which will be removed in vX.Y.Z"""
import sys
from contextlib import contextmanager
from typing import Optional

import pytest


def _soft_unimport_module(str_module):
    # once the module is imported  e.g with parsing with pytest it lives in memory
    if str_module in sys.modules:
        del sys.modules[str_module]


@contextmanager
def no_deprecated_call(match: Optional[str] = None):
    with pytest.warns(None) as record:
        yield
        try:
            w = record.pop(DeprecationWarning)
            if match is not None and match not in str(w.message):
                return
        except AssertionError:
            # no DeprecationWarning raised
            return
        raise AssertionError(f"`DeprecationWarning` was raised: {w}")
