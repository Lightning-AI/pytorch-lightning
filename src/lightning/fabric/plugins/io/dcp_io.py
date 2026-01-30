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
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Literal, Optional

import fsspec
from typing_extensions import get_args, override

from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)


CHECKPOINTER_TYPE = Literal["process", "thread", "PROCESS", "THREAD"]


class DCPIO(CheckpointIO):
    """CheckpointIO that utilizes :func:`torch.distributed.checkpoint.state_dict_saver.async_save` and
    :func:`torch.distributed.checkpoint.state_dict_loader.load` to save and load checkpoints respectively, common for
    most use cases.

    .. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

    """

    def __init__(self, checkpointer_type: CHECKPOINTER_TYPE = "process", enable_plan_caching: bool = True) -> None:
        if not _TORCH_GREATER_EQUAL_2_4:
            raise ImportError("DCPIO requires torch>=2.4.0 to use torch.distributed.checkpoint.")

        if checkpointer_type not in get_args(CHECKPOINTER_TYPE):
            raise ValueError(f"`checkpointer_type` must be one of {get_args(CHECKPOINTER_TYPE)}")

        from torch.distributed.checkpoint import DefaultSavePlanner, state_dict_saver

        super().__init__()
        self.checkpoint_future = None

        async_checkpointer_type = state_dict_saver.AsyncCheckpointerType(checkpointer_type.lower())

        # https://pytorch.org/blog/6x-faster-async-checkpointing/
        self.dcp_kwargs = {
            "async_checkpointer_type": async_checkpointer_type,
            "planner": DefaultSavePlanner(enable_plan_caching=enable_plan_caching),
        }

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
        if storage_options is not None:
            raise TypeError(
                "`Trainer.save_checkpoint(..., storage_options=...)` with `storage_options` arg"
                f" is not supported for `{self.__class__.__name__}`. Please implement your custom `CheckpointIO`"
                " to define how you'd like to use `storage_options`."
            )

        fs = get_filesystem(path)
        fs.makedirs(path, exist_ok=True)

        # waits for checkpointing to finish if one exists, avoiding queuing more then one checkpoint request at a time
        if self.checkpoint_future is not None:
            self.checkpoint_future.result()

        self.checkpoint_future = _dcp_save(checkpoint, path, dcp_kwargs=self.dcp_kwargs)

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        map_location: Optional[Callable] = lambda storage, loc: storage,
        weights_only: Optional[bool] = None,
        state_dict: Optional[dict[str, Any]] = None,
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
            state_dict: The state dict to be used during loading when using DCP. As DCP operates in place, meaning that
                the model should allocate its data first and DCP uses that storage instead.
            load_options: dict containing options to be used by `distributed.checkpoint.state_dict_loader.load

        Returns: The loaded checkpoint.

        Raises:
            FileNotFoundError: If ``path`` is not found by the ``fsspec`` filesystem

        """
        # waits for checkpointing to finish if one exists
        if self.checkpoint_future is not None:
            self.checkpoint_future.result()

        assert state_dict is not None, "When using DCPIO, `state_dict` must be provided to load the checkpoint."

        # Try to read the checkpoint at `path`. If not exist, do not restore checkpoint.
        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        return _dcp_load(path, state_dict=state_dict, dcp_kwargs=load_options)

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        """Remove checkpoint file from the filesystem.

        Args:
            path: Path to checkpoint

        """
        # waits for checkpointing to finish if one exists
        if self.checkpoint_future is not None:
            self.checkpoint_future.result()

        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug(f"Removed checkpoint: {path}")

    def teardown(self) -> None:
        """This method is called to teardown the process."""
        # waits for checkpointing to finish if one exists
        if self.checkpoint_future is not None:
            self.checkpoint_future.result()


def _dcp_save(checkpoint: dict[str, Any], filepath: _PATH, dcp_kwargs: Optional[dict[str, Any]] = None) -> Future:
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

    # only local filepaths are supported for now
    assert is_local_path(filepath), "DCP save currently only supports local filepaths."

    from torch.distributed.checkpoint import state_dict_saver

    return state_dict_saver.async_save(checkpoint, filepath, **dcp_kwargs)


def _dcp_load(
    path_or_url: _PATH,
    state_dict: dict[str, Any],
    dcp_kwargs: Optional[dict[str, Any]] = None,
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        state_dict: The state dict to be used during loading when ``use_dcp=True``.
        dcp_kwargs: Additional keyword arguments to be passed to ``torch.distributed.checkpoint.state_dict_loader.load``

    """
    if not _TORCH_GREATER_EQUAL_2_4:
        raise ImportError("Using `torch.distributed.checkpoint` for loading checkpoints requires torch>=2.4.0.")
    if state_dict is None:
        raise ValueError("When using `use_dcp=True`, `state_dict` must be provided to load the checkpoint.")
    if dcp_kwargs is None:
        dcp_kwargs = {}

    from torch.distributed.checkpoint import state_dict_loader

    if not isinstance(path_or_url, (str, Path)):
        raise ValueError("DCP loading from non-path objects is not supported.")
    if str(path_or_url).startswith(("http", "s3://", "gs://", "ftp://", "hdfs://")):
        raise ValueError("Loading checkpoints from a URL with `use_dcp=True` is not supported.")

    assert os.path.exists(path_or_url), f"Checkpoint file not found: {path_or_url}"

    return state_dict_loader.load(state_dict=state_dict, checkpoint_id=path_or_url, **dcp_kwargs)


def is_local_path(filepath: str) -> bool:
    """Check if filepath is local filesystem."""
    fs, _ = fsspec.core.url_to_fs(str(filepath))
    return fs.protocol in ("file", "local")
