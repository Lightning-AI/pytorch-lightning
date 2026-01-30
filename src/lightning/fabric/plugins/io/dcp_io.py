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
from typing import Any, Callable, Optional

from typing_extensions import override

from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.utilities.cloud_io import _atomic_save, get_filesystem
from lightning.fabric.utilities.cloud_io import _load as pl_load
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)


class DCPIO(CheckpointIO):
    """CheckpointIO that utilizes :func:`torch.distributed.checkpoint.state_dict_saver.async_save` and
    :func:`torch.distributed.checkpoint.state_dict_loader.load` to save and load checkpoints respectively, common for
    most use cases.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(self) -> None:
        if not _TORCH_GREATER_EQUAL_2_4:
            raise ImportError("DCPIO requires torch>=2.4.0 to use torch.distributed.checkpoint.")
        super().__init__()
        self._state_dict: Optional[dict[str, Any]] = None

    @property
    def state_dict(self) -> dict[str, Any]:
        """Returns the state dict saved during the last save_checkpoint call."""
        if self._state_dict is None:
            raise ValueError("No state_dict is available. Please set `state_dict` first.")
        return self._state_dict

    @state_dict.setter
    def state_dict(self, state_dict: dict[str, Any]) -> None:
        """Sets the state dict to be used during loading."""
        self._state_dict = state_dict

    @override
    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[dict[str, Any]] = None
    ) -> None:
        """Save model/training states as a checkpoint file through state-dump and file-write.

        Args:
            checkpoint: dict containing model and trainer state
            path: write-target path
            storage_options: dict containing options to be used by `distributed.checkpoint.async_save`

        """
        fs = get_filesystem(path)
        fs.makedirs(os.path.dirname(path), exist_ok=True)
        _atomic_save(checkpoint, path, use_dcp=True, dcp_kwargs=storage_options)

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        map_location: Optional[Callable] = lambda storage, loc: storage,
        weights_only: Optional[bool] = None,
        load_options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Loads checkpoint using :func:`torch.load`, with additional handling for ``fsspec`` remote loading of files.

        Args:
            path: Path to checkpoint
            map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
                locations. This argument is currently not used when loading with DCP.
            weights_only: Defaults to ``None``. If ``True``, restricts loading to ``state_dicts`` of plain
                ``torch.Tensor`` and other primitive types. If loading a checkpoint from a trusted source that contains
                an ``nn.Module``, use ``weights_only=False``. If loading checkpoint from an untrusted source, we
                recommend using ``weights_only=True``. For more information, please refer to the
                `PyTorch Developer Notes on Serialization Semantics <https://docs.pytorch.org/docs/main/notes/serialization.html#id3>`_.
                This argument is currently not used when loading with DCP.
            load_options: dict containing options to be used by `distributed.checkpoint.state_dict_loader.load

        Returns: The loaded checkpoint.

        Raises:
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem

        """
        state_dict = self.state_dict  # ensure state_dict is set before loading
        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        return pl_load(path, use_dcp=True, state_dict=state_dict, dcp_kwargs=load_options)

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        """
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug(f"Removed checkpoint: {path}")

    def teardown(self) -> None:
        """This method is called to teardown the process."""


def _dcp_save(checkpoint: dict[str, Any], filepath: _PATH, dcp_kwargs: Optional[dict[str, Any]] = None) -> None:
    """Saves a checkpoint to a given filepath using torch.distributed.checkpoint.

    Args:
        checkpoint: The object to save.
            Built to be used with the ``dump_checkpoint`` method, but can deal with anything which ``torch.save``
            accepts.
        filepath: The path to which the checkpoint will be saved.
            This points to the file that the checkpoint will be stored in.
        dcp_kwargs: Additional keyword arguments to pass to ``torch.distributed.checkpoint.state_dict_saver.async_save``
            if ``use_dcp=True``.

    """
    if not _TORCH_GREATER_EQUAL_2_4:
        raise ImportError("Using `torch.distributed.checkpoint` for saving checkpoints requires torch>=2.4.0.")
    if dcp_kwargs is None:
        dcp_kwargs = {}

    from torch.distributed.checkpoint import state_dict_saver

    state_dict_saver.async_save(checkpoint, filepath, **dcp_kwargs)
