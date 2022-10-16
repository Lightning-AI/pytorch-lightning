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
import re
from contextlib import contextmanager
from typing import Optional, Type

import pytest


@contextmanager
def no_warning_call(expected_warning: Type[Warning] = UserWarning, match: Optional[str] = None):
    with pytest.warns(None) as record:
        yield

    if match is None:
        try:
            w = record.pop(expected_warning)
        except AssertionError:
            # no warning raised
            return
    else:
        for w in record.list:
            if w.category is expected_warning and re.compile(match).search(w.message.args[0]):
                break
        else:
            return

    msg = "A warning" if expected_warning is None else f"`{expected_warning.__name__}`"
    raise AssertionError(f"{msg} was raised: {w}")
