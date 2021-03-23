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
"""Profiler to check if there are any bottlenecks in your code."""
import inspect
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union

import torch
from torch import nn, Tensor
from torch.autograd.profiler import record_function

from pytorch_lightning.profiler.profilers import BaseProfiler
from pytorch_lightning.utilities.distributed import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if TYPE_CHECKING:
    from torch.autograd.profiler import EventList
    from torch.utils.hooks import RemovableHandle

    from pytorch_lightning.core.lightning import LightningModule

log = logging.getLogger(__name__)

_PROFILER = Union[torch.autograd.profiler.profile, torch.cuda.profiler.profile, torch.autograd.profiler.emit_nvtx]


class RegisterRecordFunction:
    """
    While profiling autograd operations, this class will add labels for module names around the forward function.

    The Lightning PyTorch Profiler will activate this feature automatically. It can be deactivated as follows:

    Example::
        from pytorch_lightning.profilers import PyTorchProfiler
        profiler = PyTorchProfiler(record_module_names=False)
        Trainer(profiler=profiler)

    It can be used outside of Lightning as follows:

    Example::
        from pytorch_lightning import Trainer, seed_everything
        with RegisterRecordFunction(model):
            out = model(batch)
    """

    def __init__(self, model: nn.Module) -> None:
        self._model = model
        self._records: Dict[str, record_function] = {}
        self._handles: Dict[str, List['RemovableHandle']] = {}

    def _start_recording_forward(self, _: nn.Module, input: Tensor, record_name: str) -> Tensor:
        record = record_function(record_name)
        record.__enter__()
        self._records[record_name] = record
        return input

    def _stop_recording_forward(self, _: nn.Module, __: Tensor, output: Tensor, record_name: str) -> Tensor:
        self._records[record_name].__exit__(None, None, None)
        return output

    def __enter__(self) -> None:
        for module_name, module in self._model.named_modules():
            if module_name:
                full_name = f"{type(module).__module__}.{type(module).__name__}"
                record_name = f"{full_name}: {module_name}"
                pre_forward_handle = module.register_forward_pre_hook(
                    partial(self._start_recording_forward, record_name=record_name)
                )
                post_forward_handle = module.register_forward_hook(
                    partial(self._stop_recording_forward, record_name=record_name)
                )

                self._handles[module_name] = [pre_forward_handle, post_forward_handle]

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handles in self._handles.values():
            for h in handles:
                h.remove()
        self._handles = {}


class PyTorchProfiler(BaseProfiler):

    RECORD_FUNCTIONS = (
        "training_step_and_backward", "training_step", "backward", "validation_step", "test_step", "predict"
    )
    AVAILABLE_SORT_KEYS = (
        "cpu_time",
        "cuda_time",
        "cpu_time_total",
        "cuda_time_total",
        "cpu_memory_usage",
        "cuda_memory_usage",
        "self_cpu_memory_usage",
        "self_cuda_memory_usage",
        "count",
    )
    START_RECORD_FUNCTIONS = ('on_train_start', 'on_validation_step', 'on_test_start', 'on_predict_start')

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        path_to_export_trace: Optional[str] = None,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_functions: List[str] = None,
        record_module_names: bool = True,
        profiled_functions: Optional[List] = None,
        output_filename: Optional[str] = None,
        **profiler_kwargs: Any,
    ) -> None:
        """
        This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of
        different operators inside your model - both on the CPU and GPU

        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            group_by_input_shapes: Include operator input shapes and group calls by shape.

            emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
                Run::

                    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

                To visualize, you can either use::

                    nvvp trace_name.prof
                    torch.autograd.profiler.load_nvprof(path)

            export_to_chrome: Whether to export the sequence of profiled operators for Chrome.
                It will generate a ``.json`` file which can be read by Chrome.

            path_to_export_trace: Directory path to export ``.json`` traces when using ``export_to_chrome=True``.
                By default, it will be save where the file being is being run.

            row_limit: Limit the number of rows in a table, ``-1`` is a special value that
                removes the limit completely.

            sort_by_key: Attribute used to sort entries. By default
                they are printed in the same order as they were registered.
                Valid keys include: ``cpu_time``, ``cuda_time``, ``cpu_time_total``,
                ``cuda_time_total``, ``cpu_memory_usage``, ``cuda_memory_usage``,
                ``self_cpu_memory_usage``, ``self_cuda_memory_usage``, ``count``.

            record_functions: list of profiled functions which will create a context manager on.
                Any other will be pass through.

            record_module_names: Whether to add module names while recording autograd operation.

            profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

        Raises:
            MisconfigurationException:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``.
        """
        super().__init__(dirpath=dirpath, filename=filename, output_filename=output_filename)

        record_functions = self.__deprecation_check(profiled_functions, record_functions)

        self._group_by_input_shapes = group_by_input_shapes and profiler_kwargs.get("record_shapes", False)
        self._emit_nvtx = emit_nvtx
        self._export_to_chrome = export_to_chrome
        self._path_to_export_trace = path_to_export_trace
        self._row_limit = row_limit
        self._sort_by_key = sort_by_key or f"{'cuda' if profiler_kwargs.get('use_cuda', False) else 'cpu'}_time_total"
        self._record_functions_start = set(record_functions + list(self.START_RECORD_FUNCTIONS))
        self._record_functions = set(record_functions + list(self.RECORD_FUNCTIONS))
        self._record_module_names = record_module_names
        self._profiler_kwargs = profiler_kwargs

        self.profiler: Optional[_PROFILER] = None
        self.function_events: Optional['EventList'] = None
        self._lightning_module: Optional['LightningModule'] = None  # set by ProfilerConnector
        self._register: Optional[RegisterRecordFunction] = None
        self._parent_profiler: Optional[_PROFILER] = None
        self._recording_map: Dict[str, record_function] = {}

        if self._export_to_chrome and self._path_to_export_trace is None:
            rank_zero_warn(
                "The exported trace would be saved locally as `path_to_export_trace` is None."
                " Note: Each functions will generate its own traced file."
            )

        if self._sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise MisconfigurationException(
                f"Found sort_by_key: {self._sort_by_key}. Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

    def __deprecation_check(
        self,
        profiled_functions: Optional[List[str]],
        record_functions: Optional[List[str]],
    ) -> List[str]:
        if record_functions is None:
            record_functions = []

        if profiled_functions is not None:
            rank_zero_warn(
                "`PyTorchProfiler.profiled_functions` has been renamed to"
                " `record_functions` in v1.3 and will be removed in v1.5", DeprecationWarning
            )
            if not record_functions:
                record_functions += profiled_functions
            else:
                raise MisconfigurationException(
                    "You set `PytorchProfiler.profiled_functions` and `PyTorchProfiler.record_functions`."
                    "  Please use only the later."
                )

        return record_functions

    def setup(
        self, stage: Optional[str] = None, local_rank: Optional[int] = None, log_dir: Optional[str] = None
    ) -> None:
        super().setup(stage=stage, local_rank=local_rank, log_dir=log_dir)

        # if the user didn't provide `path_to_export_trace`,
        # set it as TensorBoardLogger log_dir if exists
        if self._path_to_export_trace is None:
            self._path_to_export_trace = log_dir

    def start(self, action_name: str) -> None:
        if self.profiler is None and action_name in self._record_functions_start:

            # close profiler if it is already opened. might happen if 2 profilers
            # are created and the first one did not call `describe`
            try:
                torch.autograd._disable_profiler()  # noqa
            except (AttributeError, RuntimeError):
                pass

            self._create_profilers()

            self.profiler.__enter__()
            if self._parent_profiler is not None:
                self._parent_profiler.__enter__()
            if self._register is not None:
                self._register.__enter__()

        if (
            self.profiler is not None and action_name in self._record_functions
            and action_name not in self._recording_map
        ):
            recording = record_function(action_name)
            recording.__enter__()
            self._recording_map[action_name] = recording

    def stop(self, action_name: str) -> None:
        if action_name in self._recording_map:
            self._recording_map[action_name].__exit__(None, None, None)
            del self._recording_map[action_name]

    def summary(self) -> str:
        if not self._profiler_kwargs.get("enabled", True) or self._emit_nvtx:
            return ""

        self._delete_profilers()

        if self._export_to_chrome:
            filename = f"{self.local_rank}_trace.json"
            path_to_trace = (
                filename if self._path_to_export_trace is None else os.path.join(self._path_to_export_trace, filename)
            )
            self.function_events.export_chrome_trace(path_to_trace)

        data = self.function_events.key_averages(group_by_input_shapes=self._group_by_input_shapes)
        table = data.table(sort_by=self._sort_by_key, row_limit=self._row_limit)

        recorded_stats = {"records": table}
        return self._stats_to_str(recorded_stats)

    def _create_profilers(self) -> None:
        if self._emit_nvtx:
            self._parent_profiler = self._create_profiler(torch.cuda.profiler.profile)
            self.profiler = self._create_profiler(torch.autograd.profiler.emit_nvtx)
        else:
            self._parent_profiler = None
            self.profiler = self._create_profiler(torch.autograd.profiler.profile)
        if self._record_module_names and self._lightning_module is not None:
            self._register = RegisterRecordFunction(self._lightning_module)

    def _create_profiler(self, profiler: Type[_PROFILER]) -> _PROFILER:
        init_parameters = inspect.signature(profiler.__init__).parameters
        kwargs = {k: v for k, v in self._profiler_kwargs.items() if k in init_parameters}
        return profiler(**kwargs)

    def _cache_functions_events(self):
        self.function_events = self.profiler.function_events

    def _delete_profilers(self) -> None:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self._cache_functions_events()
            self.profiler = None

        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None

        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None

    def teardown(self, stage: Optional[str] = None) -> None:
        self._delete_profilers()

        for k in self._recording_map:
            self.stop(k)
        self._recording_map = {}

        super().teardown()
