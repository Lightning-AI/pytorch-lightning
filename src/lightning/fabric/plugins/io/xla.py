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
from typing import Any, Callable, Dict, Optional

from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.imports import RequirementCache

from lightning.fabric.accelerators.tpu import _XLA_AVAILABLE
from lightning.fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.rank_zero import rank_zero_info
from lightning.fabric.utilities.types import _PATH


class XLACheckpointIO(TorchCheckpointIO):
    """CheckpointIO that utilizes :func:`xm.save` to save checkpoints for TPU training strategies.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.
    """

    def __init__(self, *args: Any, xser: bool = True, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(*args, **kwargs)
        self.xser = xser

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: not used in ``XLACheckpointIO.save_checkpoint``

        Raises:
            TypeError:
                If ``storage_options`` arg is passed in
        """
        if RequirementCache("omegaconf"):
            # workaround for https://github.com/pytorch/xla/issues/2773
            from omegaconf import DictConfig, ListConfig, OmegaConf

            checkpoint = apply_to_collection(checkpoint, (DictConfig, ListConfig), OmegaConf.to_container)

        if self.xser:
            import torch_xla.utils.serialization as xser

            checkpoint = xser._rewrite_data(xser._get_tensors_folder(path), checkpoint, save_tensors=True)
        else:
            import torch_xla.core.xla_model as xm

            checkpoint = xm._maybe_convert_to_cpu(checkpoint, convert=True)
        super().save_checkpoint(checkpoint, path, storage_options=storage_options)

    def load_checkpoint(
        self, path: _PATH, map_location: Optional[Callable] = lambda storage, loc: storage
    ) -> Dict[str, Any]:
        if not self.xser:
            return super().load_checkpoint(path, map_location=map_location)
        elif map_location is not None:
            # requires https://github.com/pytorch/xla/pull/4899
            rank_zero_info(f"Passing `map_location={map_location}` is not supported with `XLACheckpointIO(xser=True)`")

        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint at {path} not found. Aborting training.")

        import torch_xla.utils.serialization as xser

        return xser.load(path)
