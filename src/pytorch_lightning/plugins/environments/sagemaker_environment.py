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
import logging
import os

from pytorch_lightning.plugins.environments.lightning_environment import LightningEnvironment
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _SMDIST_AVAILABLE

if _SMDIST_AVAILABLE:
    import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401

log = logging.getLogger(__name__)


class SageMakerEnvironment(LightningEnvironment):
    """Environment for distributed training on SageMaker."""

    def __init__(self) -> None:
        if not _SMDIST_AVAILABLE:
            raise MisconfigurationException("`smdistributed` module is not available.")
        super().__init__()

    def world_size(self) -> int:
        return int(os.environ["WORLD_SIZE"])

    def set_world_size(self, size: int) -> None:
        log.debug("SageMakerEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        return int(os.environ["RANK"])

    def set_global_rank(self, rank: int) -> None:
        log.debug("SageMakerEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")
