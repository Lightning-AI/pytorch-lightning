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
"""General utilities."""

import operator
import platform
import sys

from lightning_utilities.core.imports import RequirementCache, compare_version

_NUMPY_AVAILABLE = RequirementCache("numpy")


_IS_WINDOWS = platform.system() == "Windows"

# There are two types of interactive mode we detect
# 1. The interactive Python shell: https://stackoverflow.com/a/64523765
# 2. The inspection mode via `python -i`: https://stackoverflow.com/a/6879085/1162383
_IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)

_TORCH_GREATER_EQUAL_2_2 = compare_version("torch", operator.ge, "2.2.0")
_TORCH_GREATER_EQUAL_2_3 = compare_version("torch", operator.ge, "2.3.0")
_TORCH_EQUAL_2_4_0 = compare_version("torch", operator.eq, "2.4.0")
_TORCH_GREATER_EQUAL_2_4 = compare_version("torch", operator.ge, "2.4.0")
_TORCH_GREATER_EQUAL_2_4_1 = compare_version("torch", operator.ge, "2.4.1")

_PYTHON_GREATER_EQUAL_3_10_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 10)

_UTILITIES_GREATER_EQUAL_0_10 = compare_version("lightning_utilities", operator.ge, "0.10.0")
