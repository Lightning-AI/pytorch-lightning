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
from typing import Any, Optional, Tuple, Union

from torchmetrics import PSNR as _PSNR

from pytorch_lightning.metrics.utils import deprecated_metrics, void


class PSNR(_PSNR):
    @deprecated_metrics(target=_PSNR)
    def __init__(
        self,
        data_range: Optional[float] = None,
        base: float = 10.0,
        reduction: str = "elementwise_mean",
        dim: Optional[Union[int, Tuple[int, ...]]] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        """
        This implementation refers to :class:`~torchmetrics.PSNR`.

        .. deprecated::
            Use :class:`~torchmetrics.PSNR`. Will be removed in v1.5.0.
        """
        void(data_range, base, reduction, dim, compute_on_step, dist_sync_on_step, process_group)
