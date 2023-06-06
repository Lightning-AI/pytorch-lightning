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

from lightning_utilities.core.imports import compare_version

_IS_WINDOWS = platform.system() == "Windows"

# There are two types of interactive mode we detect
# 1. The interactive Python shell: https://stackoverflow.com/a/64523765
# 2. The inspection mode via `python -i`: https://stackoverflow.com/a/6879085/1162383
_IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)

_TORCH_GREATER_EQUAL_1_12 = compare_version("torch", operator.ge, "1.12.0")
_TORCH_GREATER_EQUAL_1_13 = compare_version("torch", operator.ge, "1.13.0")
_TORCH_GREATER_EQUAL_2_0 = compare_version("torch", operator.ge, "2.0.0", use_base_version=True)
_TORCH_GREATER_EQUAL_2_1 = compare_version("torch", operator.ge, "2.1.0", use_base_version=True)
_TORCH_EQUAL_2_0 = _TORCH_GREATER_EQUAL_2_0 and not _TORCH_GREATER_EQUAL_2_1

_PYTHON_GREATER_EQUAL_3_8_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 8)
_PYTHON_GREATER_EQUAL_3_10_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 10)
