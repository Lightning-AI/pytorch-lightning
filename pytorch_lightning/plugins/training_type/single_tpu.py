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
import os
from typing import Any, Dict

from pytorch_lightning.core.decorators import parameter_validation
from pytorch_lightning.plugins.training_type.single_device import SingleDevicePlugin
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE, _TPU_AVAILABLE
from pytorch_lightning.utilities.apply_func import apply_to_collection

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm

if _OMEGACONF_AVAILABLE:
    from omegaconf import DictConfig, ListConfig, OmegaConf


class SingleTPUPlugin(SingleDevicePlugin):
    """Plugin for training on a single TPU device."""

    def __init__(self, device: int, debug: bool = False):

        device = xm.xla_device(device)
        super().__init__(device)

        self.debug = debug
        self.tpu_local_core_rank = 0
        self.tpu_global_core_rank = 0

    @property
    def is_distributed(self) -> bool:
        return False

    @parameter_validation
    def model_to_device(self) -> None:
        self.model.to(self.root_device)

    def pre_dispatch(self) -> None:
        if isinstance(self.device, int):
            self.device = xm.xla_device(self.device)

        if self.debug:
            os.environ["PT_XLA_DEBUG"] = str(1)

        self.tpu_local_core_rank = xm.get_local_ordinal()
        self.tpu_global_core_rank = xm.get_ordinal()

    def save(self, state_dict: Dict, path: str) -> None:
        xm.save(state_dict, path)

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: str) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            filepath: write-target file's path
        """
        # Related Issue: https://github.com/pytorch/xla/issues/2773
        if _OMEGACONF_AVAILABLE:
            checkpoint = apply_to_collection(checkpoint, (DictConfig, ListConfig), OmegaConf.to_container)
        self.save({k: v for k, v in checkpoint.items() if k != "callbacks"}, filepath)

    def teardown(self) -> None:
        # TPU teardown
        os.environ.pop("PT_XLA_DEBUG", None)
