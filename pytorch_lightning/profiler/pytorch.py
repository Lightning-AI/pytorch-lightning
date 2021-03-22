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
from typing import Any, Dict, List, Optional, Type, Union

import torch
from torch import nn, Tensor
from torch.autograd.profiler import EventList, record_function

from pytorch_lightning.profiler.profilers import BaseProfiler
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.distributed import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)

_PROFILER = Union[torch.autograd.profiler.profile, torch.cuda.profiler.profile, torch.autograd.profiler.emit_nvtx]


class RegisterRecordFunction:
    """
    While profiling autograd operations, this class will add label with module name
    around the forward function.
    The Lightning PyTorch Profiler will activate this feature automatically.
    It can be deactivated as follows:
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

    def __init__(self, model: nn.Module):
        self._model = model
        self._records = {}
        self.handles = {}

    def _start_recording_forward(self, module: nn.Module, input: Tensor, record_name: str):
        record = record_function(record_name)
        record.__enter__()
        self._records[record_name] = record
        return input

    def _stop_recording_forward(self, module: nn.Module, input: Tensor, output: Tensor, record_name: str):
        self._records[record_name].__exit__(None, None, None)
        return output

    def __enter__(self):
        for module_name, module in self._model.named_modules():
            if module_name != '':
                full_name = type(module).__module__ + '.' + type(module).__name__
                record_name = f"{full_name}: {module_name}"
                pre_forward_handle = module.register_forward_pre_hook(
                    partial(self._start_recording_forward, record_name=record_name)
                )
                post_forward_handle = module.register_forward_hook(
                    partial(self._stop_recording_forward, record_name=record_name)
                )

                self.handles[module_name] = [pre_forward_handle, post_forward_handle]

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        for handles in self.handles.values():
            for h in handles:
                h.remove()


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
        output_filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = True,
        path_to_export_trace: Optional[str] = None,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_functions: List[str] = None,
        local_rank: Optional[int] = None,
        profiled_functions: List[str] = None,
        record_module_names: bool = True,
        **profiler_kwargs: Any,
    ) -> None:
        """
        This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of
        different operators inside your model - both on the CPU and GPU

        Args:
            output_filename: optionally save profile results to file instead of printing
                to std out when training is finished. When using ``ddp``,
                each rank will stream the profiled operation to their own file
                with the extension ``_{rank}.txt``

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

            local_rank: When running in distributed setting, local_rank is used for each process
                to write to their own file if `output_fname` is provided.

            profiler_kwargs: Keyword arguments for the PyTorch profiler. This depends on your PyTorch version

            record_module_names: Whether to add module names while recording autograd operation.

        Raises:
            MisconfigurationException:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``, or
                if log file is not a ``.txt`` file.
            ValueError:
                If you attempt to stop recording an action which was never started.
        """
        if output_filename is not None and not output_filename.endswith(".txt"):
            raise MisconfigurationException("`output_filename` should be a `.txt` file.")

        record_functions = self.__deprecation_check(profiled_functions, record_functions)

        self.output_fname = output_filename
        self.record_functions = set(record_functions + list(self.RECORD_FUNCTIONS))
        self.sort_by_key = sort_by_key or f"{'cuda' if profiler_kwargs.get('use_cuda', False) else 'cpu'}_time_total"
        self.group_by_input_shapes = group_by_input_shapes and profiler_kwargs.get("record_shapes", False)
        self.row_limit = row_limit
        self.emit_nvtx = emit_nvtx
        self.export_to_chrome = export_to_chrome
        self.path_to_export_trace = path_to_export_trace
        self.record_module_names = record_module_names
        self.lightning_module = None  # set by ProfilerConnector
        self.register = None
        self.profiler_kwargs = profiler_kwargs
        self.profiler = None
        self._parent_profiler = None

        if self.export_to_chrome and self.path_to_export_trace is None:
            rank_zero_warn(
                "The exported trace would be saved locally as `path_to_export_trace` is None."
                " Note: Each functions will generate its own traced file."
            )

        if self.sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise MisconfigurationException(
                f"Found sort_by_key: {self.sort_by_key}. Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

        self.recording_map: Dict[str, record_function] = {}
        self.profiler: Optional[_PROFILER] = None
        self.function_events: Optional[EventList] = None
        self._profiler_instantiated: bool = False

        super().__init__(output_filename=output_filename, local_rank=local_rank)

    def __deprecation_check(self, profiled_functions: List[str] = [], record_functions: List[str] = []) -> List[str]:
        if record_functions is None:
            record_functions = []

        if profiled_functions is not None:
            rank_zero_warn(
                "`PyTorchProfiler.profiled_functions` has been renamed to"
                " `record_functions` in v1.3 and will be removed in v1.5", DeprecationWarning
            )
            if (len(record_functions) == 0 or len(profiled_functions) == 0):
                record_functions += profiled_functions
            else:
                raise MisconfigurationException(
                    "You set `PytorchProfiler.profiled_functions` and `PyTorchProfiler.record_functions`."
                    "  Please use only the later."
                )
        if record_functions is None:
            record_functions = []

        return record_functions

    def on_train_start(self, local_rank: Optional[int] = None, log_dir: Optional[str] = None) -> None:
        super().on_train_start(local_rank=local_rank, log_dir=log_dir)

        # if the user didn't provide `path_to_export_trace`,
        # set it as TensorBoardLogger log_dir if exists
        if self.path_to_export_trace is None:
            self.path_to_export_trace = log_dir

        # when logging to `log.info`, only perform profiling on rank 0
        if local_rank is not None and local_rank > 0 and self.output_fname is None:
            self._rank_zero_only_wrap()

    def _rank_zero_only_wrap(self) -> None:
        self.start = rank_zero_only(self.start)
        self.stop = rank_zero_only(self.stop)
        self.summary = rank_zero_only(self.summary)
        self.describe = rank_zero_only(self.describe)

    def start(self, action_name: str) -> None:
        if not self._profiler_instantiated and action_name in (
            list(self.START_RECORD_FUNCTIONS) + list(self.record_functions)
        ):

            # close profiler if it is already opened
            try:
                torch.autograd._disable_profiler()
            except (AttributeError, RuntimeError):
                pass

            self._create_profilers()

            self.profiler.__enter__()
            if self._parent_profiler is not None:
                self._parent_profiler.__enter__()

            self._profiler_instantiated = True

            if self.record_module_names and self.lightning_module is not None:
                self.register = RegisterRecordFunction(self.lightning_module)
                self.register.__enter__()

        if (
            self._profiler_instantiated and action_name in self.record_functions
            and action_name not in self.recording_map
        ):
            recording = record_function(action_name)
            recording.__enter__()
            self.recording_map[action_name] = recording

    def stop(self, action_name: str) -> None:
        if action_name in self.recording_map:
            self.recording_map[action_name].__exit__(None, None, None)
            del self.recording_map[action_name]

    def summary(self) -> str:
        if not self.profiler_kwargs.get("enabled", True):
            return ""

        local_rank = 0 if self.local_rank is None else self.local_rank

        self.profiler.__exit__(None, None, None)
        if not self.emit_nvtx:
            self.function_events = self.profiler.function_events
        self.profiler = None
        self._profiler_instantiated = False

        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None

        if self.register is not None:
            self.register.__exit__(None, None, None)

        if self.emit_nvtx:
            return ""

        if self.export_to_chrome:
            filename = f"{local_rank}_trace.json"
            path_to_trace = (
                filename if self.path_to_export_trace is None else os.path.join(self.path_to_export_trace, filename)
            )
            self.function_events.export_chrome_trace(path_to_trace)

        data = self.function_events.key_averages(group_by_input_shapes=self.group_by_input_shapes)
        table = data.table(sort_by=self.sort_by_key, row_limit=self.row_limit)

        recorded_stats = {}
        recorded_stats["records"] = table

        return self.stats_to_str(recorded_stats)

    def _create_profilers(self) -> None:
        if self.emit_nvtx:
            self._parent_profiler = self._create_profiler(torch.cuda.profiler.profile)
            self.profiler = self._create_profiler(torch.autograd.profiler.emit_nvtx)
        else:
            self._parent_profiler = None
            self.profiler = self._create_profiler(torch.autograd.profiler.profile)

    def _create_profiler(self, profiler: Type[_PROFILER]) -> _PROFILER:
        init_parameters = inspect.signature(profiler.__init__).parameters
        kwargs = {k: v for k, v in self.profiler_kwargs.items() if k in init_parameters}
        return profiler(**kwargs)

    def teardown(self):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)

        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None

        if self.register is not None:
            self.register.__exit__(None, None, None)

        for record in self.recording_map.values():
            record.__exit__(None, None, None)

        super().teardown()
