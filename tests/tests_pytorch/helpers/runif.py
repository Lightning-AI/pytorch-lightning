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
import pytest

from lightning.fabric.utilities.imports import _IS_WINDOWS
from lightning.pytorch.utilities.imports import _TORCH_EQUAL_2_8, _TORCH_EQUAL_2_9
from lightning.pytorch.utilities.testing import _runif_reasons


def RunIf(**kwargs):
    reasons, marker_kwargs = _runif_reasons(**kwargs)
    return pytest.mark.skipif(condition=len(reasons) > 0, reason=f"Requires: [{' + '.join(reasons)}]", **marker_kwargs)


# todo: RuntimeError: makeDeviceForHostname(): unsupported gloo device
_xfail_gloo_windows = pytest.mark.xfail(
    RuntimeError,
    strict=True,
    condition=(_IS_WINDOWS and (_TORCH_EQUAL_2_8 or _TORCH_EQUAL_2_9)),
    reason="makeDeviceForHostname(): unsupported gloo device",
)
