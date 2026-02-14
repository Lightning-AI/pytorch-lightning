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
from concurrent.futures import Future
from typing import Any, Literal, Optional

import fsspec
from typing_extensions import get_args, override

from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_4
from lightning.fabric.utilities.types import _PATH

log = logging.getLogger(__name__)

CHECKPOINTER_TYPE = Literal["process", "thread"]


class AsyncCheckpointIO(CheckpointIO):
    """Experimental asynchronous CheckpointIO backed by torch.distributed.checkpoint.

    Notes:
        - Only supports saving/loading `state_dict`
        - Currently supports only local filesystem paths.
        - Loading is in-place: caller must provide a pre-allocated `state_dict`

    """

    def __init__(
        self,
        checkpointer_type: CHECKPOINTER_TYPE = "process",
        enable_plan_caching: bool = True,
        save_options: Optional[dict[str, Any]] = None,
        load_options: Optional[dict[str, Any]] = None,
    ) -> None:
        if not _TORCH_GREATER_EQUAL_2_4:
            raise ImportError("AsyncCheckpointIO requires torch>=2.4.0.")

        from torch.distributed.checkpoint import DefaultSavePlanner, state_dict_saver

        super().__init__()

        checkpointer_type = checkpointer_type.lower()
        if checkpointer_type not in get_args(CHECKPOINTER_TYPE):
            raise ValueError(f"`checkpointer_type` must be one of {get_args(CHECKPOINTER_TYPE)}")

        self.checkpoint_future: Optional[Future] = None

        async_type = state_dict_saver.AsyncCheckpointerType(checkpointer_type)

        # https://pytorch.org/blog/6x-faster-async-checkpointing/
        # https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/
        default_save_options = {
            "async_checkpointer_type": async_type,
            "planner": DefaultSavePlanner(enable_plan_caching=enable_plan_caching),
        }
        self.save_options = {**default_save_options, **(save_options or {})}
        self.load_options = dict(load_options or {})

    def _wait(self) -> None:
        if self.checkpoint_future is None:
            return
        try:
            self.checkpoint_future.result()
        except Exception as ex:
            raise RuntimeError("AsyncCheckpointIO checkpointing failed.") from ex
        finally:
            self.checkpoint_future = None

    @override
    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[dict[str, Any]] = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "`storage_options` is not supported by AsyncCheckpointIO. Implement a custom CheckpointIO if needed."
            )

        self._wait()

        local_path_checks(path)

        fs = get_filesystem(path)
        fs.makedirs(path, exist_ok=True)

        self.checkpoint_future = _dcp_save(state_dict=checkpoint, filepath=path, dcp_kwargs=self.save_options)

    @override
    def load_checkpoint(
        self,
        path: _PATH,
        *,
        state: Optional[dict[str, Any]] = None,
        map_location: Optional[Any] = None,
        weights_only: Optional[bool] = None,
    ) -> Optional[dict[str, Any]]:
        if map_location is not None:
            raise TypeError(
                "`map_location` is not supported by AsyncCheckpointIO. "
                "Device placement is determined by the provided state_dict layouts."
            )

        self._wait()

        if state is None:
            raise ValueError("When using AsyncCheckpointIO, `state` must be provided for in-place loading.")

        _dcp_load(path, state_dict=state, dcp_kwargs=self.load_options)

        # Return None per CheckpointIO contract to indicate the checkpoint was fully restored in-place.
        return None

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
        raise ImportError("AsyncCheckpointIO requires torch>=2.4.0.")

    local_path_checks(filepath)

    from torch.distributed.checkpoint import state_dict_saver

    return state_dict_saver.async_save(state_dict, filepath, **(dcp_kwargs or {}))


def _dcp_load(
    path_or_url: _PATH,
    state_dict: dict[str, Any],
    dcp_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    if not _TORCH_GREATER_EQUAL_2_4:
        raise ImportError("AsyncCheckpointIO requires torch>=2.4.0.")

    local_path_checks(path_or_url, check_if_exists=True)

    from torch.distributed.checkpoint import state_dict_loader

    state_dict_loader.load(
        state_dict=state_dict,
        checkpoint_id=path_or_url,
        **(dcp_kwargs or {}),
    )


# TODO: Replace with remote filesystem support once async DCP readers support non-local paths.
def local_path_checks(path: _PATH, check_if_exists: bool = False) -> None:
    fs, _ = fsspec.core.url_to_fs(str(path))

    protocol = fs.protocol
    if isinstance(protocol, (tuple, list)):
        # With fsspec, protocol can be a tuple in some implementations.
        is_local = "file" in protocol or "local" in protocol
    else:
        is_local = protocol in ("file", "local")

    if not is_local:
        raise ValueError(f"AsyncCheckpointIO only supports local filesystem paths, but got: {path}")

    if check_if_exists and not fs.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
