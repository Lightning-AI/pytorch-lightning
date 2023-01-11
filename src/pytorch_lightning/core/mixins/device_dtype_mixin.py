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

from lightning_fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from pytorch_lightning.utilities import rank_zero_deprecation


class DeviceDtypeModuleMixin(_DeviceDtypeModuleMixin):
    def __init__(self) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.core.mixins.DeviceDtypeModuleMixin` has been deprecated in v1.8.0 and will be"
            " removed in v2.0.0. This class is internal but you can copy over its implementation."
        )
        super().__init__()
