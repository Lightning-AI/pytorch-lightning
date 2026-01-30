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
    """Experimental CheckpointIO backed by torch.distributed.checkpoint.

    Notes:
        - Only supports saving/loading `state_dict`
        - Only supports local filesystem paths
        - Loading is in-place: caller must provide a pre-allocated `state_dict`

    """

    def __init__(self, checkpointer_type: CHECKPOINTER_TYPE = "process", enable_plan_caching: bool = True) -> None:
        if not _TORCH_GREATER_EQUAL_2_4:
            raise ImportError("DCPIO requires torch>=2.4.0.")

        if checkpointer_type not in get_args(CHECKPOINTER_TYPE):
            raise ValueError(f"`checkpointer_type` must be one of {get_args(CHECKPOINTER_TYPE)}")

        from torch.distributed.checkpoint import DefaultSavePlanner, state_dict_saver

        super().__init__()
        self.checkpoint_future: Optional[Future] = None

        checkpointer_type = checkpointer_type.lower()
        async_type = state_dict_saver.AsyncCheckpointerType(checkpointer_type)

        self.dcp_kwargs = {
            "async_checkpointer_type": async_type,
            "planner": DefaultSavePlanner(enable_plan_caching=enable_plan_caching),
        }

    def _wait(self) -> None:
        if self.checkpoint_future is not None:
            try:
                self.checkpoint_future.result()
            except Exception as ex:
                raise RuntimeError("Async DCP checkpointing failed.") from ex

    @override
    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[dict[str, Any]] = None
    ) -> None:
        if storage_options is not None:
            raise TypeError("`storage_options` is not supported by DCPIO. Implement a custom CheckpointIO if needed.")

        if not is_local_path(path):
            raise ValueError("DCPIO only supports local filesystem paths.")

        self._wait()

        fs = get_filesystem(path)
        fs.makedirs(path, exist_ok=True)

        self.checkpoint_future = _dcp_save(state_dict=checkpoint, filepath=path, dcp_kwargs=self.dcp_kwargs)

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        map_location: Optional[Callable] = lambda storage, loc: storage,
        weights_only: Optional[bool] = None,
        state_dict: Optional[dict[str, Any]] = None,
        load_options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        self._wait()

        if state_dict is None:
            raise ValueError("When using DCPIO, `state_dict` must be provided for in-place loading.")

        if not is_local_path(path):
            raise ValueError("DCPIO only supports local filesystem paths.")

        fs = get_filesystem(path)
        if not fs.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        _dcp_load(path, state_dict=state_dict, dcp_kwargs=load_options)

        # Lightning expects a checkpoint dict
        return {"state_dict": state_dict}

    @override
    def remove_checkpoint(self, path: _PATH) -> None:
        self._wait()

        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            log.debug(f"Removed checkpoint: {path}")

    def teardown(self) -> None:
        self._wait()


def _dcp_save(
    state_dict: dict[str, Any],
    filepath: _PATH,
    dcp_kwargs: Optional[dict[str, Any]] = None,
) -> Future:
    if not _TORCH_GREATER_EQUAL_2_4:
        raise ImportError("torch>=2.4.0 required for DCP.")

    if not is_local_path(filepath):
        raise ValueError("DCP save only supports local filesystem paths.")

    from torch.distributed.checkpoint import state_dict_saver

    return state_dict_saver.async_save(state_dict, filepath, **(dcp_kwargs or {}))


def _dcp_load(
    path_or_url: _PATH,
    state_dict: dict[str, Any],
    dcp_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    if not _TORCH_GREATER_EQUAL_2_4:
        raise ImportError("torch>=2.4.0 required for DCP.")

    if not isinstance(path_or_url, (str, Path)):
        raise ValueError("DCP loading only supports filesystem paths.")

    if str(path_or_url).startswith(("http://", "https://", "s3://", "gs://", "ftp://", "hdfs://")):
        raise ValueError("Remote paths are not supported by DCPIO.")

    if not os.path.exists(path_or_url):
        raise FileNotFoundError(f"Checkpoint not found: {path_or_url}")

    from torch.distributed.checkpoint import state_dict_loader

    state_dict_loader.load(
        state_dict=state_dict,
        checkpoint_id=path_or_url,
        **(dcp_kwargs or {}),
    )


def is_local_path(filepath: _PATH) -> bool:
    fs, _ = fsspec.core.url_to_fs(str(filepath))
    return fs.protocol in ("file", "local")
