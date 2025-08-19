# Copyright The Lightning AI team.
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
from typing import Any, Optional

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import RequirementCache
from typing_extensions import override

from lightning.fabric.accelerators.xla import _XLA_AVAILABLE
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)


class XLACheckpointIO(TorchCheckpointIO):
    """CheckpointIO that utilizes ``xm.save`` to save checkpoints for TPU training strategies.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(*args, **kwargs)

    @override
    def save_checkpoint(self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``XLACheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in

        """
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        if RequirementCache("omegaconf"):
            # workaround for https://github.com/pytorch/xla/issues/2773
            from omegaconf import DictConfig, ListConfig, OmegaConf

            checkpoint = apply_to_collection(checkpoint, (DictConfig, ListConfig), OmegaConf.to_container)
        import torch_xla.core.xla_model as xm

        cpu_data = xm._maybe_convert_to_cpu(checkpoint, convert=True)
        log.debug(f"Saving checkpoint: {path}")
        torch.save(cpu_data, path)
