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
"""General utilities."""
import operator
import platform
import sys

from lightning_utilities.core.imports import compare_version

_IS_WINDOWS = platform.system() == "Windows"
_IS_INTERACTIVE = hasattr(sys, "ps1")  # https://stackoverflow.com/a/64523765
_PYTHON_GREATER_EQUAL_3_8_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 8)
_PYTHON_GREATER_EQUAL_3_10_0 = (sys.version_info.major, sys.version_info.minor) >= (3, 10)
_TORCH_GREATER_EQUAL_1_9_1 = compare_version("torch", operator.ge, "1.9.1")
_TORCH_GREATER_EQUAL_1_10 = compare_version("torch", operator.ge, "1.10.0")
_TORCH_LESSER_EQUAL_1_10_2 = compare_version("torch", operator.le, "1.10.2")
_TORCH_GREATER_EQUAL_1_11 = compare_version("torch", operator.ge, "1.11.0")
_TORCH_GREATER_EQUAL_1_12 = compare_version("torch", operator.ge, "1.12.0")
_TORCH_GREATER_EQUAL_1_13 = compare_version("torch", operator.ge, "1.13.0", use_base_version=True)
