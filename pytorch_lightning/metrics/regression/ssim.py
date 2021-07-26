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
from typing import Any, Optional, Sequence

from torchmetrics import SSIM as _SSIM

from pytorch_lightning.metrics.utils import deprecated_metrics, void


class SSIM(_SSIM):
    @deprecated_metrics(target=_SSIM)
    def __init__(
        self,
        kernel_size: Sequence[int] = (11, 11),
        sigma: Sequence[float] = (1.5, 1.5),
        reduction: str = "elementwise_mean",
        data_range: Optional[float] = None,
        k1: float = 0.01,
        k2: float = 0.03,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        This implementation refers to :class:`~torchmetrics.SSIM`.

        .. deprecated::
            Use :class:`~torchmetrics.SSIM`. Will be removed in v1.5.0.
        """
        void(kernel_size, sigma, reduction, data_range, k1, k2, compute_on_step, dist_sync_on_step, process_group)
