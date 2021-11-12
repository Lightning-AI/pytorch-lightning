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
"""Test deprecated functionality which will be removed in v1.8.0."""
import pytest

from tests.deprecated_api import _soft_unimport_module


def test_v1_8_0_deprecated_env_vars_connector_defaults_from_env_vars():
    _soft_unimport_module("pytorch_lightning.trainer.connectors.env_vars_connector._defaults_from_env_vars")
    with pytest.deprecated_call(match="deprecated in v1.6, and will be removed in v1.8."):
        from pytorch_lightning.trainer.connectors.env_vars_connector import _defaults_from_env_vars  # noqa: F401
