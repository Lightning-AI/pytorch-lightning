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

from abc import ABC, abstractmethod

import torch
from torch.nn import Module


class LayerSync(ABC):
    """Abstract base class for creating plugins that wrap layers of a model with synchronization logic for
    multiprocessing."""

    @abstractmethod
    def apply(self, model: Module) -> Module:
        """Override this method to apply synchronization to the layers of this model."""

    @abstractmethod
    def revert(self, model: Module) -> Module:
        """Override this method to undo all modifications made in :meth:`apply`."""


class NativeSyncBatchNorm(LayerSync):
    """A plugin that wraps all batch normalization layers of a model with synchronization logic for
    multiprocessing.

    This plugin has no effect in single-device operation.
    """

    def apply(self, model: Module) -> Module:
        """Add global batchnorm for a model spread across multiple GPUs and nodes.

        Override this method to synchronize batchnorm layers between specific process groups instead
        of the whole world.

        Args:
            model: Reference to the current LightningModule

        Return:
            LightningModule with batchnorm layers synchronized within the process groups.
        """
        return torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    def revert(self, model: Module) -> Module:
        """Convert the wrapped batchnorm layers back to regular batchnorm layers.

        Args:
            model: Reference to the current LightningModule

        Return:
            LightningModule with regular batchnorm layers that will no longer sync across processes.
        """
        # Code adapted from https://github.com/pytorch/pytorch/issues/41081#issuecomment-783961547
        # Original author: Kapil Yedidi (@kapily)
        converted_module = model
        if isinstance(model, torch.nn.modules.batchnorm.SyncBatchNorm):
            # Unfortunately, LayerSync does not store the original class - if it did
            # we could return the one that was originally created.
            converted_module = _BatchNormXd(
                model.num_features, model.eps, model.momentum, model.affine, model.track_running_stats
            )
            if model.affine:
                with torch.no_grad():
                    converted_module.weight = model.weight
                    converted_module.bias = model.bias
            converted_module.running_mean = model.running_mean
            converted_module.running_var = model.running_var
            converted_module.num_batches_tracked = model.num_batches_tracked
            if hasattr(model, "qconfig"):
                converted_module.qconfig = model.qconfig
        for name, child in model.named_children():
            converted_module.add_module(name, self.revert(child))
        del model
        return converted_module


class _BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input: torch.Tensor) -> None:
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the subclass.
        # Here, we are bypassing some tensor sanity checks and trusting that the user
        # provides the right input dimensions at inference.
        return
