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
"""Warning-related utilities."""

from lightning_lite.utilities.rank_zero import rank_zero_deprecation
from lightning_lite.utilities.warnings import PossibleUserWarning  # noqa: F401
from lightning_lite.utilities.warnings import WarningCache as NewWarningCache


class WarningCache(NewWarningCache):
    def __init__(self) -> None:
        rank_zero_deprecation(
            "`pytorch_lightning.utilities.warnings.WarningCache` has been deprecated in v1.8.0 and will be"
            " removed in v1.10.0. Please use `lightning_lite.utilities.warnings.WarningCache` instead."
        )
        super().__init__()
