"""Mirroring deprecated API"""

from abc import ABC

from pytorch_lightning.utilities import rank_zero_warn


class TrainerDeprecatedAPITillVer0_9(ABC):

    def __init__(self):
        super().__init__()  # mixin calls super too

    @property
    def show_progress_bar(self):
        """Back compatibility, will be removed in v0.9.0"""
        rank_zero_warn("Argument `show_progress_bar` is now set by `progress_bar_refresh_rate` since v0.7.2"
                       " and this method will be removed in v0.9.0", DeprecationWarning)
        return self.progress_bar_callback and self.progress_bar_callback.refresh_rate >= 1

    @show_progress_bar.setter
    def show_progress_bar(self, tf):
        """Back compatibility, will be removed in v0.9.0"""
        rank_zero_warn("Argument `show_progress_bar` is now set by `progress_bar_refresh_rate` since v0.7.2"
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
        rank_zero_warn("Argument `num_tpu_cores` is now set by `tpu_cores` since v0.7.6"
                       " and this argument will be removed in v0.9.0", DeprecationWarning)
