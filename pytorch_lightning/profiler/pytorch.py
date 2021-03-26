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
from typing import List, Optional

import torch

from pytorch_lightning.profiler.profilers import BaseProfiler
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.distributed import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException

log = logging.getLogger(__name__)


class PyTorchProfiler(BaseProfiler):

    PROFILED_FUNCTIONS = ("training_step_and_backward", "validation_step", "test_step")
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
        enabled: bool = True,
        use_cuda: bool = False,
        record_shapes: bool = False,
        profile_memory: bool = False,
        group_by_input_shapes: bool = False,
        with_stack: bool = False,
        use_kineto: bool = False,
        use_cpu: bool = True,
        emit_nvtx: bool = False,
        export_to_chrome: bool = False,
        path_to_export_trace: str = None,
        row_limit: int = 20,
        sort_by_key: Optional[str] = None,
        profiled_functions: Optional[List] = None,
        local_rank: Optional[int] = None,
    ):
        """
        This profiler uses PyTorch's Autograd Profiler and lets you inspect the cost of
        different operators inside your model - both on the CPU and GPU

        Args:

            output_filename: optionally save profile results to file instead of printing
                to std out when training is finished. When using ``ddp``,
                each rank will stream the profiled operation to their own file
                with the extension ``_{rank}.txt``

            enabled: Setting this to False makes this context manager a no-op.

            use_cuda: Enables timing of CUDA events as well using the cudaEvent API.
                Adds approximately 4us of overhead to each tensor operation.

            record_shapes: If shapes recording is set, information about input dimensions will be collected.

            profile_memory: Whether to report memory usage, default: True (Introduced in PyTorch 1.6.0)

            group_by_input_shapes: Include operator input shapes and group calls by shape.

            with_stack: record source information (file and line number) for the ops (Introduced in PyTorch 1.7.0)

            use_kineto: experimental support for Kineto profiler (Introduced in PyTorch 1.8.0)

            use_cpu: use_kineto=True and can be used to lower the overhead
                for GPU-only profiling (Introduced in PyTorch 1.8.0)

            emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
                Run::

                    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

                To visualize, you can either use::

                    nvvp trace_name.prof
                    torch.autograd.profiler.load_nvprof(path)

            export_to_chrome: Wether to export the sequence of profiled operators for Chrome.
                It will generate a ``.json`` file which can be read by Chrome.

            path_to_export_trace: Directory path to export ``.json`` traces when using ``export_to_chrome=True``.
                By default, it will be save where the file being is being run.

            row_limit: Limit the number of rows in a table, `0` is a special value that
                removes the limit completely.

            sort_by_key: Keys to sort out profiled table

            profiled_functions: list of profiled functions which will create a context manager on.
                Any other will be pass through.

            local_rank: When running in distributed setting, local_rank is used for each process
                to write to their own file if `output_fname` is provided.

        Raises:
            MisconfigurationException:
                If arg ``sort_by_key`` is not present in ``AVAILABLE_SORT_KEYS``, or
                if log file is not a ``.txt`` file.
            ValueError:
                If you attempt to stop recording an action which was never started.
        """

        self.profiled_actions = {}
        self.enabled = enabled
        self.profiled_functions = profiled_functions or self.PROFILED_FUNCTIONS
        self.use_cuda = use_cuda
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.sort_by_key = sort_by_key or ("cuda_time_total" if self.use_cuda else "cpu_time_total")
        self.with_stack = with_stack
        self.group_by_input_shapes = group_by_input_shapes and record_shapes
        self.use_kineto = use_kineto
        self.use_cpu = use_cpu
        self.row_limit = row_limit
        self.emit_nvtx = emit_nvtx
        self.export_to_chrome = export_to_chrome
        self.path_to_export_trace = path_to_export_trace

        if export_to_chrome and path_to_export_trace is None:
            rank_zero_warn(
                "The exported trace would be save locally as `path_to_export_trace` is empty."
                " Note: Each functions will generate its own traced file."
            )

        if self.sort_by_key not in self.AVAILABLE_SORT_KEYS:
            raise MisconfigurationException(
                f"Found sort_by_key: {sort_by_key}. Should be within {self.AVAILABLE_SORT_KEYS}. "
            )

        self.profiled_actions = {}
        self.context_names = {}
        self.running_stack = []
        self.profiler = None

        self.output_fname = output_filename
        self.output_file = None
        if local_rank is not None:
            self.on_train_start(local_rank=local_rank)
            self.on_train_start = super().on_train_start

    def on_train_start(self, local_rank: Optional[str] = None):
        self.local_rank = local_rank

        # when logging to `log.info`, only perform profiling on rank 0
        if local_rank != 0 and self.output_fname is None:
            self.wrap_functions_into_rank_zero_only()

        if self.output_fname:
            if local_rank is not None:
                if '.txt' not in self.output_fname:
                    raise MisconfigurationException("Log file should be .txt file.")

                self.output_fname = self.output_fname.replace(".txt", f"_{self.local_rank}.txt")

            fs = get_filesystem(self.output_fname)
            self.output_file = fs.open(self.output_fname, "w")

        streaming_out = [self.output_file.write] if self.output_file else [log.info]
        super().__init__(output_streams=streaming_out)

    def wrap_functions_into_rank_zero_only(self):
        self.start = rank_zero_only(self.start)
        self.stop = rank_zero_only(self.stop)
        self.summary = rank_zero_only(self.summary)
        self.describe = rank_zero_only(self.describe)

    def start(self, action_name: str) -> None:
        if action_name not in self.profiled_functions:
            return

        if len(self.running_stack) > 0:
            self._stop(self.running_stack[-1])
        self.running_stack.append(action_name)

        self.context_names[action_name] = "/".join(self.running_stack)

        self._start(action_name)

    def _start(self, action_name: str) -> None:
        if self.emit_nvtx:
            self._parent_profiler = self._create_profiler(action_name, torch.cuda.profiler.profile, enter=True)
            self._create_profiler(action_name, torch.autograd.profiler.emit_nvtx)
        else:
            self._create_profiler(action_name, torch.autograd.profiler.profile)

    def _create_profiler(self, action_name, profiler, enter=True):
        init_args = inspect.signature(profiler.__init__).parameters
        profiler_args = {k: v for k, v in vars(self).items() if k in init_args}
        pr = profiler(**profiler_args)
        if enter:
            out_pr = pr.__enter__()
            if out_pr is not None:
                pr = out_pr
        self.profiler = pr
        return self.profiler

    def _stop(self, action_name: str) -> None:
        if self.profiler is None:
            return

        self.profiler.__exit__(exc_type=None, exc_val=None, exc_tb=None)

        if isinstance(self.profiler, torch.autograd.profiler.emit_nvtx):
            # when running ``emit_nvtx``, PyTorch requires 2 context manager.
            # The parent_profiler is being closed too.
            self._parent_profiler.__exit__(None, None, None)
            return

        function_events = self.profiler.function_events
        self.profiler = None
        for name in self.running_stack:
            if name not in self.profiled_actions:
                self.profiled_actions[name] = function_events
            else:
                self.profiled_actions[name] += function_events

    def stop(self, action_name: str) -> None:
        if action_name not in self.profiled_functions:
            return

        if len(self.running_stack) == 0 or self.running_stack[-1] != action_name:
            raise ValueError(  # pragma: no-cover
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        self._stop(action_name)
        self.running_stack.pop()
        # restore running profiler
        if len(self.running_stack) > 0:
            self._start(self.running_stack[-1])

    def summary(self) -> str:
        recorded_stats = {}
        output_string = ''
        local_rank = '0' if self.local_rank is None else self.local_rank

        if not self.enabled:
            return output_string

        for action_name, function_events in self.profiled_actions.items():

            # next line is a workaround for a pytorch issue (fixed on master, still present
            # on 1.7). Without it the code fails with `AssertionError: There is already a CPU
            # parent event for detach`
            function_events.populate_cpu_children = lambda: None

            if self.export_to_chrome:
                filename = f"{action_name}_{local_rank}_trace.json"
                path_to_trace = filename if self.path_to_export_trace is None \
                    else os.path.join(self.path_to_export_trace, filename)
                function_events.export_chrome_trace(path_to_trace)

            if self.emit_nvtx:
                return output_string

            else:
                data = function_events.key_averages(group_by_input_shapes=self.group_by_input_shapes)
                table = data.table(sort_by=self.sort_by_key, row_limit=self.row_limit)
                recorded_stats[action_name] = table

        # log to standard out
        output_string = f"{os.linesep}Profiler Report{os.linesep}"
        for action, stats in recorded_stats.items():
            output_string += (f"{os.linesep}Profile stats for: {action} rank: {local_rank} {os.linesep}{stats}")

        return output_string

    def describe(self):
        """Logs a profile report after the conclusion of the training run."""
        super().describe()
        if self.output_file:
            self.output_file.flush()

    def __del__(self):
        """Close profiler's stream."""
        if self.output_file:
            self.output_file.close()
