"""Mirroring deprecated API"""

from abc import ABC

from pytorch_lightning.utilities import rank_zero_warn


class TrainerDeprecatedAPITillVer0_8(ABC):

    def __init__(self):
        super().__init__()  # mixin calls super too

    @property
    def nb_gpu_nodes(self):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `nb_gpu_nodes` has renamed to `num_nodes` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.num_nodes

    @property
    def num_gpu_nodes(self):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `num_gpu_nodes` has renamed to `num_nodes` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.num_nodes

    @num_gpu_nodes.setter
    def num_gpu_nodes(self, num_nodes):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `num_gpu_nodes` has renamed to `num_nodes` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        self.num_nodes = num_nodes

    @property
    def gradient_clip(self):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `gradient_clip` has renamed to `gradient_clip_val` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.gradient_clip_val

    @gradient_clip.setter
    def gradient_clip(self, gradient_clip):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `gradient_clip` has renamed to `gradient_clip_val` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        self.gradient_clip_val = gradient_clip

    @property
    def max_nb_epochs(self):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `max_nb_epochs` has renamed to `max_epochs` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.max_epochs

    @max_nb_epochs.setter
    def max_nb_epochs(self, max_epochs):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `max_nb_epochs` has renamed to `max_epochs` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        self.max_epochs = max_epochs

    @property
    def min_nb_epochs(self):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `min_nb_epochs` has renamed to `min_epochs` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.min_epochs

    @min_nb_epochs.setter
    def min_nb_epochs(self, min_epochs):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `min_nb_epochs` has renamed to `min_epochs` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        self.min_epochs = min_epochs

    @property
    def nb_sanity_val_steps(self):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `nb_sanity_val_steps` has renamed to "
                       "`num_sanity_val_steps` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.num_sanity_val_steps

    @nb_sanity_val_steps.setter
    def nb_sanity_val_steps(self, nb):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `nb_sanity_val_steps` has renamed to "
                       "`num_sanity_val_steps` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        self.num_sanity_val_steps = nb

    @property
    def default_save_path(self):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `default_save_path` has renamed to `default_root_dir` since v0.5.x"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.default_root_dir

    @default_save_path.setter
    def default_save_path(self, path):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("Attribute `default_save_path` has renamed to `default_root_dir` since v0.5.x"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        self.default_root_dir = path

    @property
    def tng_tqdm_dic(self):
        """Back compatibility, will be removed in v0.8.0"""
        rank_zero_warn("`tng_tqdm_dic` has renamed to `training_tqdm_dict` since v0.5.0"
                       " and this method will be removed in v0.8.0", DeprecationWarning)
        return self.progress_bar_dict


class TrainerDeprecatedAPITillVer0_9(ABC):

    def __init__(self):
        super().__init__()  # mixin calls super too

    @property
    def show_progress_bar(self):
        """Back compatibility, will be removed in v0.9.0"""
        rank_zero_warn("Argument `show_progress_bar` is now set by `progress_bar_refresh_rate` since v0.7.2"
                       " and this method will be removed in v0.9.0", DeprecationWarning)
        return self.progress_bar_refresh_rate >= 1

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
