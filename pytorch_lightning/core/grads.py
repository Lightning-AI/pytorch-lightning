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
"""
Module to describe gradients. This class is deprecated in v1.3 and will be removed in v1.5
"""
from typing import Dict, Union

from torch.nn import Module

from pytorch_lightning.utilities.distributed import rank_zero_deprecation
from pytorch_lightning.utilities.grads import grad_norm as new_grad_norm


class GradInformation(Module):

    def grad_norm(self, norm_type: Union[float, int, str]) -> Dict[str, float]:
        """Compute each parameter's gradient's norm and their overall norm.

        .. deprecated:: v1.3
            Will be removed in v1.5.0. Use :func:`pytorch_lightning.utilities.grads.grad_norm` instead.
        """
        rank_zero_deprecation(
            "LightningModule.grad_norm is deprecated in v1.3 and will be removed in v1.5."
            " Use grad_norm from pytorch_lightning.utilities.grads instead."
        )
        return new_grad_norm(self, norm_type)
