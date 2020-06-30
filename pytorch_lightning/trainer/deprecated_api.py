"""Mirroring deprecated API"""

from abc import ABC
from typing import Union

from pytorch_lightning.utilities import rank_zero_warn


class TrainerDeprecatedAPITillVer0_9(ABC):
    progress_bar_dict: ...
    progress_bar_callback: ...

    def __init__(self):
        super().__init__()  # mixin calls super too

    @property
    def show_progress_bar(self):
        """Back compatibility, will be removed in v0.9.0"""
        rank_zero_warn("Attribute `show_progress_bar` is now set by `progress_bar_refresh_rate` since v0.7.2"
                       " and this method will be removed in v0.9.0", DeprecationWarning)
        return self.progress_bar_callback and self.progress_bar_callback.refresh_rate >= 1

    @show_progress_bar.setter
    def show_progress_bar(self, tf):
        """Back compatibility, will be removed in v0.9.0"""
        rank_zero_warn("Attribute `show_progress_bar` is now set by `progress_bar_refresh_rate` since v0.7.2"
                       " and this method will be removed in v0.9.0", DeprecationWarning)

    @property
    def training_tqdm_dict(self):
        """Back compatibility, will be removed in v0.9.0"""
        rank_zero_warn("`training_tqdm_dict` was renamed to `progress_bar_dict` in v0.7.3"
                       " and this method will be removed in v0.9.0", DeprecationWarning)
        return self.progress_bar_dict

    @property
    def num_tpu_cores(self):
        """Back compatibility, will be removed in v0.9.0"""
        rank_zero_warn("Attribute `num_tpu_cores` is now set by `tpu_cores` since v0.7.6"
                       " and this argument will be removed in v0.9.0", DeprecationWarning)


class TrainerDeprecatedAPITillVer0_10(ABC):
    limit_val_batches: Union[int, float]
    limit_test_batches: Union[int, float]
    limit_train_batches: Union[int, float]
    overfit_batches: Union[int, float]

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
