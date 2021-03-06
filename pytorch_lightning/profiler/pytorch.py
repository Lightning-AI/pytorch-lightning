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
from typing import List, Optional, Union, Type, Any, Tuple, Dict

import torch
from torch.autograd.profiler import record_function, EventList, parse_event_records

from pytorch_lightning.profiler.profilers import BaseProfiler
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.distributed import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)

_PROFILER = Union[torch.autograd.profiler.profile, torch.cuda.profiler.profile, torch.autograd.profiler.emit_nvtx]


class PyTorchProfiler(BaseProfiler):

    RECORD_FUNCTIONS = ("training_step_and_backward", "training_step", "backward", "validation_step", "test_step")
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

    def __init__(
        self,
        output_filename: Optional[str] = None,
        group_by_input_shapes: bool = False,
        emit_nvtx: bool = False,
        export_to_chrome: bool = False,
        path_to_export_trace: Optional[str] = None,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        record_functions: Optional[List[str]] = None,
        local_rank: Optional[int] = None,
        profiled_functions: Optional[List[str]] = None,
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

            row_limit: Limit the number of rows in a table, ``0`` is a special value that
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
        self.record_functions = record_functions or self.RECORD_FUNCTIONS
        self.sort_by_key = sort_by_key or f"{'cuda' if profiler_kwargs.get('use_cuda', False) else 'cpu'}_time_total"
        self.group_by_input_shapes = group_by_input_shapes and profiler_kwargs.get("record_shapes", False)
        self.row_limit = row_limit
        self.emit_nvtx = emit_nvtx
        self.export_to_chrome = export_to_chrome
        self.path_to_export_trace = path_to_export_trace
        self.profiler_kwargs = profiler_kwargs

        if self.export_to_chrome and self.path_to_export_trace is None:
            rank_zero_warn(
                "The exported trace would be saved locally as `path_to_export_trace` is None."
                " Note: Each functions will generate its own traced file."
            )

        if self.sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise MisconfigurationException(
                f"Found sort_by_key: {self.sort_by_key}. Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

        self.stack: List[Tuple[str, record_function]] = []
        self.profiler: Optional[_PROFILER] = None
        self.function_events: Dict[str, EventList] = {}

        super().__init__(local_rank=local_rank)

    def __deprecation_check(
        self,
        profiled_functions: Optional[List[str]],
        record_functions: Optional[List[str]]
    ) -> Optional[List[str]]:
        if profiled_functions is not None:
            rank_zero_warn(
                "`PyTorchProfiler.profiled_functions` has been renamed to"
                " `record_functions` in v1.3 and will be removed in v1.5", DeprecationWarning
            )
            if record_functions is None:
                record_functions = profiled_functions
            else:
                raise MisconfigurationException(
                    "You set `PytorchProfiler.profiled_functions` and `PyTorchProfiler.record_functions`."
                    "  Please use only the later."
                )
        return record_functions

    def on_train_start(self, local_rank: Optional[int] = None, log_dir: Optional[str] = None) -> None:
        super().on_train_start(local_rank=local_rank, log_dir=log_dir)

        # if the user didn't `path_to_export_trace`,
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
        if action_name not in self.record_functions:
            return

        if self.profiler is None:
            self._create_profilers()

            self.profiler.__enter__()
            if self._parent_profiler is not None:
                self._parent_profiler.__enter__()

        recording = record_function(action_name)
        self.stack.append((action_name, recording))
        recording.__enter__()

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

    def stop(self, action_name: str) -> None:
        if self.profiler is None or action_name not in self.record_functions:
            return

        if not self.stack or self.stack[-1][0] != action_name:
            raise ValueError(f"Attempting to stop recording an action ({action_name}) which was never started.")

        action_name, recording = self.stack.pop()
        recording.__exit__(None, None, None)

        self.function_events[action_name] = self.thing()
        self.profiler.function_events = None

        if not self.stack:
            self.profiler.__exit__(None, None, None)
            if self._parent_profiler is not None:
                self._parent_profiler.__exit__(None, None, None)

    def thing(self):
        """TODO: Adapted from ..."""
        profiler = self.profiler
        kind = torch.autograd.ProfilerState.CUDA if profiler.use_cuda else torch.autograd.ProfilerState.CPU
        config = torch.autograd.ProfilerConfig(
            kind,
            profiler.record_shapes,
            profiler.profile_memory,
            profiler.with_stack
        )

        records = torch.autograd._disable_profiler()

        function_events = EventList(
            parse_event_records(records),
            use_cuda=profiler.use_cuda,
            profile_memory=profiler.profile_memory
        )
        if profiler.with_stack:
            function_events.set_backward_stacktraces()

        torch.autograd._enable_profiler(config)

        return function_events

    def summary(self) -> str:
        if not self.profiler_kwargs.get("enabled", True) or self.emit_nvtx:
            return ""

        local_rank = 0 if self.local_rank is None else self.local_rank
        recorded_stats = {}
        function_events = self.profiler.function_events

        # next line is a workaround for a pytorch issue (fixed on master, still present
        # on 1.7). Without it the code fails with `AssertionError: There is already a CPU
        # parent event for detach`
        function_events.populate_cpu_children = lambda: None

        if self.export_to_chrome:
            filename = f"{function_events.name}_{local_rank}_trace.json"
            path_to_trace = (
                filename
                if self.path_to_export_trace is None else os.path.join(self.path_to_export_trace, filename)
            )
            function_events.export_chrome_trace(path_to_trace)

        data = function_events.key_averages(group_by_input_shapes=self.group_by_input_shapes)
        table = data.table(sort_by=self.sort_by_key, row_limit=self.row_limit)
        recorded_stats[action_name] = table

        return self.stats_to_str(recorded_stats)

    def __del__(self) -> None:
        super().__del__()
        self.profiler.__exit__(None, None, None)
        self._parent_profiler.__exit__(None, None, None)
