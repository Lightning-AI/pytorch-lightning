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

from lightning.fabric.utilities.apply_func import move_data_to_device
from lightning.fabric.utilities.data import suggested_max_num_workers
from lightning.fabric.utilities.distributed import is_shared_filesystem
from lightning.fabric.utilities.enums import LightningEnum
from lightning.fabric.utilities.rank_zero import (
    rank_zero_deprecation,
    rank_zero_info,
    rank_zero_only,
    rank_zero_warn,
)
from lightning.fabric.utilities.speed_monitor import SpeedMonitor, measure_flops
from lightning.fabric.utilities.warnings import disable_possible_user_warnings

__all__ = [
    "move_data_to_device",
    "suggested_max_num_workers",
    "is_shared_filesystem",
    "LightningEnum",
    "rank_zero_info",
    "rank_zero_deprecation",
    "rank_zero_only",
    "rank_zero_warn",
    "disable_possible_user_warnings",
    "measure_flops",
    "SpeedMonitor",
]
