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
from typing import Any, Callable, Union

from torch.optim import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.accelerators.accelerator import Accelerator
from pytorch_lightning.plugins.precision import MixedPrecisionPlugin
from pytorch_lightning.plugins.training_type.single_tpu import SingleTPUPlugin
from pytorch_lightning.plugins.training_type.tpu_spawn import TPUSpawnPlugin
from pytorch_lightning.utilities import _XLA_AVAILABLE, GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _XLA_AVAILABLE:
    import torch_xla.core.xla_model as xm
    from torch_xla._patched_functions import clip_grad_norm_

    # rename to mock in a test
    xla_clip_grad_norm_ = clip_grad_norm_


class TPUAccelerator(Accelerator):
    """ Accelerator for TPU devices. """

    def setup(self, trainer: 'pl.Trainer', model: 'pl.LightningModule') -> None:
        """
        Raises:
            MisconfigurationException:
                If AMP is used with TPU, or if TPUs are not using a single TPU core or TPU spawn training.
        """
        if isinstance(self.precision_plugin, MixedPrecisionPlugin):
            raise MisconfigurationException(
                "amp + tpu is not supported. "
                "Only bfloats are supported on TPU. Consider using TPUHalfPrecisionPlugin"
            )

        if not isinstance(self.training_type_plugin, (SingleTPUPlugin, TPUSpawnPlugin)):
            raise MisconfigurationException("TPUs only support a single tpu core or tpu spawn training.")
        return super().setup(trainer, model)

    def run_optimizer_step(
        self, optimizer: Optimizer, optimizer_idx: int, lambda_closure: Callable, **kwargs: Any
    ) -> None:
        xm.optimizer_step(optimizer, optimizer_args={'closure': lambda_closure, **kwargs})

    def clip_gradients(
        self,
        optimizer: Optimizer,
        clip_val: Union[float, int],
        gradient_clip_algorithm: GradClipAlgorithmType = GradClipAlgorithmType.NORM,
    ) -> None:
        assert gradient_clip_algorithm == GradClipAlgorithmType.NORM, \
            "Only NORM gradient clipping is supported on TPU for now"

        grad_clip_val = float(clip_val)
        if grad_clip_val <= 0:
            return

        parameters = self.model.parameters()
        norm_type = 2.0

        xla_clip_grad_norm_(parameters, grad_clip_val, norm_type)
