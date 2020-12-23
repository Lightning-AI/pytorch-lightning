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
import pytest

from tests.deprecated_api import _soft_unimport_module


def test_v1_4_0_deprecated_imports():
    _soft_unimport_module('pytorch_lightning.utilities.argparse_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.argparse_utils import from_argparse_args  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.model_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.model_utils import is_overridden  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.warning_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.warning_utils import WarningCache  # noqa: F811 F401

    _soft_unimport_module('pytorch_lightning.utilities.xla_device_utils')
    with pytest.deprecated_call(match='will be removed in v1.4'):
        from pytorch_lightning.utilities.xla_device_utils import XLADeviceUtils  # noqa: F811 F401
