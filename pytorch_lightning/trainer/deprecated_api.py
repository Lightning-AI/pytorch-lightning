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

"""Mirroring deprecated API"""
from abc import ABC

from pytorch_lightning.utilities import rank_zero_warn


class TrainerDeprecatedAPITillVer0_11(ABC):
    flush_logs_every_n_steps: int
    log_every_n_steps: int

    def __init__(self):
        super().__init__()  # mixin calls super too

    @property
    def log_save_interval(self) -> int:
        """Back compatibility, will be removed in v0.11.0"""
        rank_zero_warn("Attribute `log_save_interval` is now set by `flush_logs_every_n_steps` since v0.10.0"
                       " and this method will be removed in v0.11.0", DeprecationWarning)
        return self.flush_logs_every_n_steps

    @log_save_interval.setter
    def log_save_interval(self, val: int):
        """Back compatibility, will be removed in v0.11.0"""
        rank_zero_warn("Attribute `log_save_interval` is now set by `flush_logs_every_n_steps` since v0.10.0"
                       " and this method will be removed in v0.11.0", DeprecationWarning)
        self.flush_logs_every_n_steps = val

    @property
    def row_log_interval(self) -> int:
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `row_log_interval` is now set by `log_every_n_steps` since v0.10.0"
                       " and this method will be removed in v0.11.0", DeprecationWarning)
        return self.log_every_n_steps

    @row_log_interval.setter
    def row_log_interval(self, val: int):
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `row_log_interval` is now set by `log_every_n_steps` since v0.10.0"
                       " and this method will be removed in v0.11.0", DeprecationWarning)
        self.log_every_n_steps = val
