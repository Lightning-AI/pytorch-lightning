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
from typing import Union

from pytorch_lightning.utilities import rank_zero_warn


class TrainerDeprecatedAPITillVer0_10(ABC):
    limit_val_batches: Union[int, float]
    limit_test_batches: Union[int, float]
    limit_train_batches: Union[int, float]
    overfit_batches: Union[int, float]
    is_global_zero: bool
    _weights_save_path: str
    weights_save_path: str

    def __init__(self):
        super().__init__()  # mixin calls super too

    @property
    def val_percent_check(self) -> Union[int, float]:
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `val_percent_check` is now set by `limit_val_batches` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        return self.limit_val_batches

    @val_percent_check.setter
    def val_percent_check(self, pct):
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `val_percent_check` is now set by `limit_val_batches` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        self.limit_val_batches = pct

    @property
    def test_percent_check(self) -> Union[int, float]:
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `test_percent_check` is now set by `limit_test_batches` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        return self.limit_test_batches

    @test_percent_check.setter
    def test_percent_check(self, pct):
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `test_percent_check` is now set by `limit_test_batches` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        self.limit_test_batches = pct

    @property
    def train_percent_check(self) -> Union[int, float]:
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `train_percent_check` is now set by `limit_train_batches` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        return self.limit_train_batches

    @train_percent_check.setter
    def train_percent_check(self, pct):
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `train_percent_check` is now set by `limit_train_batches` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        self.limit_train_batches = pct

    @property
    def overfit_pct(self) -> Union[int, float]:
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `train_percent_check` is now set by `overfit_batches` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        return self.overfit_batches

    @overfit_pct.setter
    def overfit_pct(self, pct):
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `train_percent_check` is now set by `overfit_batches` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        self.overfit_batches = pct

    @property
    def proc_rank(self) -> int:
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `proc_rank` is now set by `global_rank` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        return self.global_rank

    @proc_rank.setter
    def proc_rank(self, rank):
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `proc_rank` is now set by `global_rank` since v0.8.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        self.global_rank = rank

    @property
    def ckpt_path(self) -> str:
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `ckpt_path` is now set by `weights_save_path` since v0.9.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        return self.weights_save_path if self.is_global_zero else None

    @ckpt_path.setter
    def ckpt_path(self, path: str):
        """Back compatibility, will be removed in v0.10.0"""
        rank_zero_warn("Attribute `ckpt_path` is now set by `weights_save_path` since v0.9.0"
                       " and this method will be removed in v0.10.0", DeprecationWarning)
        self._weights_save_path = path
