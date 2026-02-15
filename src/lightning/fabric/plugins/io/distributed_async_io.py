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
import warnings
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

import torch.distributed as dist
from typing_extensions import get_args, override

from lightning.fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_2_4,
    _TORCH_GREATER_EQUAL_2_7,
    _TORCH_GREATER_EQUAL_2_9,
)
from lightning.fabric.utilities.types import _PATH

if TYPE_CHECKING:
    from torch.distributed.checkpoint import AsyncSaveResponse

log = logging.getLogger(__name__)

CHECKPOINTER_TYPE = Literal["process", "thread"]


class DistributedAsyncCheckpointIO(CheckpointIO):
    """Experimental asynchronous CheckpointIO backed by torch.distributed.checkpoint.

    Notes:
        - Loading is in-place: caller must provide a pre-allocated `state_dict`

    """

    def __init__(
        self,
        checkpointer_type: Optional[CHECKPOINTER_TYPE] = None,
        no_dist: bool = True,
        enable_plan_caching: bool = True,
        save_options: Optional[dict[str, Any]] = None,
        load_options: Optional[dict[str, Any]] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize the asynchronous checkpoint I/O plugin.

        Args:
            checkpointer_type: The async executor type used by
                ``torch.distributed.checkpoint``. Can be ``"thread"`` or ``"process"``.
                If ``None``, the executor is selected automatically:

                - ``"thread"`` when ``torch.distributed`` is unavailable or not initialized
                - ``"process"`` when running in a distributed environment

                Thread mode is suitable for single-device training, while process mode
                enables distributed async checkpointing.

            no_dist: If True, this function will assume the intent is to save a checkpoint on a single rank/process.

            enable_plan_caching: Whether to enable planner caching in
                :class:`torch.distributed.checkpoint.DefaultSavePlanner`, which can
                reduce overhead when saving repeatedly with the same state structure.

            save_options: Optional keyword arguments forwarded to
                ``torch.distributed.checkpoint.state_dict_saver.async_save``.
                User-provided options override the defaults configured by this plugin.

            load_options: Optional keyword arguments forwarded to
                ``torch.distributed.checkpoint.state_dict_loader.load`` during loading.

        Raises:
            ImportError: If ``torch<2.4.0`` is installed.

        """
        if not _TORCH_GREATER_EQUAL_2_4:
            raise ImportError("AsyncCheckpointIO requires torch>=2.4.0.")

        from torch.distributed.checkpoint import DefaultSavePlanner, state_dict_saver

        super().__init__()
        no_dist = no_dist or (not dist.is_available()) or (not dist.is_initialized())
        self.timeout = timeout

        if checkpointer_type is None:
            checkpointer_type = "thread" if no_dist else "process"

        checkpointer_type = checkpointer_type.lower()
        if checkpointer_type not in get_args(CHECKPOINTER_TYPE):
            raise ValueError(f"`checkpointer_type` must be one of {get_args(CHECKPOINTER_TYPE)}")

        self._checkpointer_type = checkpointer_type
        self.checkpoint_future: Optional[Union[Future, AsyncSaveResponse]] = None

        # https://pytorch.org/blog/6x-faster-async-checkpointing/
        # https://pytorch.org/blog/distributed-checkpoint-efficient-checkpointing-in-large-scale-jobs/
        default_save_options: dict[str, Any] = {}
        if _TORCH_GREATER_EQUAL_2_9:
            default_save_options["no_dist"] = no_dist
        if _TORCH_GREATER_EQUAL_2_7:
            async_type = state_dict_saver.AsyncCheckpointerType(self._checkpointer_type)
            default_save_options["async_checkpointer_type"] = async_type
            default_save_options["planner"] = DefaultSavePlanner(enable_plan_caching=enable_plan_caching)
        print(f"{default_save_options=}")
        self.save_options = {**default_save_options, **(save_options or {})}
        self.load_options = dict(load_options or {})
        self._disable_safe_warnings()

    def _disable_safe_warnings(self) -> None:
        """Disable the suppression of warnings that are known to be emitted by torch.distributed.checkpoint."""
        _safe_warnings = [
            "TypedStorage is deprecated",
            "torch.distributed is disabled, unavailable or uninitialized",
        ]

        for pattern in _safe_warnings:
            warnings.filterwarnings("ignore", message=pattern)

    def _wait(self) -> None:
        if self.checkpoint_future is None:
            return
        try:
            self.checkpoint_future.result(timeout=self.timeout)
        except Exception as ex:
            raise RuntimeError("AsyncCheckpointIO checkpointing failed.") from ex

    @override
    @property
    def _requires_state_conversion(self) -> bool:
        return True

    @property
    @override
    def requires_cpu_collectives(self) -> bool:
        """Async checkpointing requires CPU collectives to be available.

        The process-based async executor in ``torch.distributed.checkpoint`` relies on
        a CPU-capable process group to coordinate metadata exchange and background
        save operations. When running with a CUDA-only backend (e.g., ``nccl``),
        async checkpointing will fail unless a CPU backend such as ``gloo`` is also
        enabled (e.g., ``"cpu:gloo,cuda:nccl"``).

        """
        return True

    @override
    def save_checkpoint(
        self, checkpoint: dict[str, Any], path: _PATH, storage_options: Optional[dict[str, Any]] = None
    ) -> None:
        if storage_options is not None:
            raise TypeError(
                "`storage_options` is not supported by AsyncCheckpointIO. Implement a custom CheckpointIO if needed."
            )

        self._wait()

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
    ) -> dict[str, Any]:
        if map_location is not None:
            raise TypeError(
                "`map_location` is not supported by AsyncCheckpointIO. "
                "Device placement is determined by the provided state_dict layouts."
            )

        self._wait()

        if state is None:
            raise ValueError("When using AsyncCheckpointIO, `state` must be provided for in-place loading.")

        _dcp_load(path, state_dict=state, dcp_kwargs=self.load_options)

        # Return empty dict to indicate the checkpoint was fully restored in-place.
        return {}

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
) -> Union[Future, "AsyncSaveResponse"]:
    if not _TORCH_GREATER_EQUAL_2_4:
        raise ImportError("AsyncCheckpointIO requires torch>=2.4.0.")

    from torch.distributed.checkpoint import state_dict_saver

    return state_dict_saver.async_save(state_dict=state_dict, checkpoint_id=filepath, **(dcp_kwargs or {}))


def _dcp_load(
    path_or_url: _PATH,
    state_dict: dict[str, Any],
    dcp_kwargs: Optional[dict[str, Any]] = None,
) -> None:
    if not _TORCH_GREATER_EQUAL_2_4:
        raise ImportError("AsyncCheckpointIO requires torch>=2.4.0.")

    from torch.distributed.checkpoint import state_dict_loader

    state_dict_loader.load(
        state_dict=state_dict,
        checkpoint_id=path_or_url,
        **(dcp_kwargs or {}),
    )
