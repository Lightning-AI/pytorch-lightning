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

import cProfile
import inspect
import io
import os
import pstats
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from typing import Optional, Union

import numpy as np
import torch

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class BaseProfiler(ABC):
    """
    If you wish to write a custom profiler, you should inhereit from this class.
    """

    def __init__(self, output_streams: Optional[Union[list, tuple]] = None):
        """
        Args:
            output_streams: callable
        """
        if output_streams:
            if not isinstance(output_streams, (list, tuple)):
                output_streams = [output_streams]
        else:
            output_streams = []
        self.write_streams = output_streams

    @abstractmethod
    def start(self, action_name: str) -> None:
        """Defines how to start recording an action."""

    @abstractmethod
    def stop(self, action_name: str) -> None:
        """Defines how to record the duration once an action is complete."""

    @contextmanager
    def profile(self, action_name: str) -> None:
        """
        Yields a context manager to encapsulate the scope of a profiled action.

        Example::

            with self.profile('load training data'):
                # load training data code

        The profiler will start once you've entered the context and will automatically
        stop once you exit the code block.
        """
        try:
            self.start(action_name)
            yield action_name
        finally:
            self.stop(action_name)

    def profile_iterable(self, iterable, action_name: str) -> None:
        iterator = iter(iterable)
        while True:
            try:
                self.start(action_name)
                value = next(iterator)
                self.stop(action_name)
                yield value
            except StopIteration:
                self.stop(action_name)
                break

    def describe(self) -> None:
        """Logs a profile report after the conclusion of the training run."""
        for write in self.write_streams:
            write(self.summary())

    @abstractmethod
    def summary(self) -> str:
        """Create profiler summary in text format."""


class PassThroughProfiler(BaseProfiler):
    """
    This class should be used when you don't want the (small) overhead of profiling.
    The Trainer uses this class by default.
    """

    def __init__(self):
        super().__init__(output_streams=None)

    def start(self, action_name: str) -> None:
        pass

    def stop(self, action_name: str) -> None:
        pass

    def summary(self) -> str:
        return ""


class SimpleProfiler(BaseProfiler):
    """
    This profiler simply records the duration of actions (in seconds) and reports
    the mean duration of each action and the total time spent over the entire training run.
    """

    def __init__(self, output_filename: Optional[str] = None, extended=True):
        """
        Args:
            output_filename: optionally save profile results to file instead of printing
                to std out when training is finished.
        """
        self.current_actions = {}
        self.recorded_durations = defaultdict(list)
        self.extended = extended

        self.output_fname = output_filename
        self.output_file = None
        if self.output_fname:
            fs = get_filesystem(self.output_fname)
            self.output_file = fs.open(self.output_fname, "w")

        streaming_out = [self.output_file.write] if self.output_file else [log.info]
        self.start_time = time.monotonic()
        super().__init__(output_streams=streaming_out)

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def make_report(self):
        total_duration = time.monotonic() - self.start_time
        report = [[a, d, 100. * np.sum(d) / total_duration] for a, d in self.recorded_durations.items()]
        report.sort(key=lambda x: x[2], reverse=True)
        return report, total_duration

    def summary(self) -> str:
        output_string = "\n\nProfiler Report\n"

        if self.extended:

            if len(self.recorded_durations) > 0:
                max_key = np.max([len(k) for k in self.recorded_durations.keys()])

                def log_row(action, mean, num_calls, total, per):
                    row = f"{os.linesep}{action:<{max_key}s}\t|  {mean:<15}\t|"
                    row += f"{num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                    return row

                output_string += log_row("Action", "Mean duration (s)", "Num calls", "Total time (s)", "Percentage %")
                output_string_len = len(output_string)
                output_string += f"{os.linesep}{'-' * output_string_len}"
                report, total_duration = self.make_report()
                output_string += log_row("Total", "-", "_", f"{total_duration:.5}", "100 %")
                output_string += f"{os.linesep}{'-' * output_string_len}"
                for action, durations, duration_per in report:
                    output_string += log_row(
                        action, f"{np.mean(durations):.5}", f"{len(durations):}",
                        f"{np.sum(durations):.5}", f"{duration_per:.5}"
                    )
        else:
            def log_row(action, mean, total):
                return f"{os.linesep}{action:<20s}\t|  {mean:<15}\t|  {total:<15}"

            output_string += log_row("Action", "Mean duration (s)", "Total time (s)")
            output_string += f"{os.linesep}{'-' * 65}"

            for action, durations in self.recorded_durations.items():
                output_string += log_row(
                    action, f"{np.mean(durations):.5}", f"{np.sum(durations):.5}"
                )
        output_string += os.linesep
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


class AdvancedProfiler(BaseProfiler):
    """
    This profiler uses Python's cProfiler to record more detailed information about
    time spent in each function call recorded during a given action. The output is quite
    verbose and you should only use this if you want very detailed reports.
    """

    def __init__(self, output_filename: Optional[str] = None, line_count_restriction: float = 1.0):
        """
        Args:
            output_filename: optionally save profile results to file instead of printing
                to std out when training is finished.
            line_count_restriction: this can be used to limit the number of functions
                reported for each action. either an integer (to select a count of lines),
                or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)
        """
        self.profiled_actions = {}
        self.line_count_restriction = line_count_restriction

        self.output_fname = output_filename
        self.output_file = None
        if self.output_fname:
            fs = get_filesystem(self.output_fname)
            self.output_file = fs.open(self.output_fname, "w")

        streaming_out = [self.output_file.write] if self.output_file else [log.info]
        super().__init__(output_streams=streaming_out)

    def start(self, action_name: str) -> None:
        if action_name not in self.profiled_actions:
            self.profiled_actions[action_name] = cProfile.Profile()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name: str) -> None:
        pr = self.profiled_actions.get(action_name)
        if pr is None:
            raise ValueError(  # pragma: no-cover
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        pr.disable()

    def summary(self) -> str:
        recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
            ps.print_stats(self.line_count_restriction)
            recorded_stats[action_name] = s.getvalue()

        # log to standard out
        output_string = f"{os.linesep}Profiler Report{os.linesep}"
        for action, stats in recorded_stats.items():
            output_string += (
                f"{os.linesep}Profile stats for: {action}{os.linesep}{stats}"
            )

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


class PytorchProfiler(BaseProfiler):
    """
    This profiler uses PyTorch's Autograd Profiler and let's you inspect the cost of
    different operators inside your model - both on the CPU and GPU
    """

    PROFILED_FUNCTIONS = ["training_step", "validation_step", "test_step"]

    def __init__(self,
                 output_filename: Optional[str] = None,
                 enabled=True,
                 use_cuda=False,
                 record_shapes=False,
                 profile_memory=False,
                 group_by_input_shape=False,
                 with_stack=False,
                 use_kineto=False,
                 use_cpu=False,
                 emit_nvtx=False,
                 export_to_chrome=False,
                 path_to_export_trace=None,
                 sort_by_key: str = "cpu_time_total"):
        """
        Args:
            output_filename: optionally save profile results to file instead of printing
                to std out when training is finished.
            enabled: Setting this to False makes this context manager a no-op. Default: True
            use_cuda: Enables timing of CUDA events as well using the cudaEvent API.
                Adds approximately 4us of overhead to each tensor operation. Default: True
            record_shapes:  If shapes recording is set, information about input dimensions will be collected.
            profile_memory: Whether to report memory usage, default: True (1.6.0)
            with_stack: record source information (file and line number) for the ops (1.7.0)
            use_kineto: experimental support for Kineto profiler (1.8.0)
            use_cpu: use_kineto=True and can be used to lower the overhead for GPU-only profiling (1.8.0)
            emit_nvtx: Context manager that makes every autograd operation emit an NVTX range
                * Run: nvprof --profile-from-start off -o trace_name.prof -- <regular command here>
                To visualize, you can either use:
                    * nvvp trace_name.prof
                    * torch.autograd.profiler.load_nvprof(path)
            export_to_chrome: Wether to export the sequence of profiled operators for Chrome.
            sort_by_key: Keys to sort out profiled table
            path_to_export_trace: Path to exported traces. By default, it will be save
                where the file being is being run.
        """
        self.profiled_actions = {}
        # PyTorch Profiler doesn't seem to work with multiple processes
        enabled = enabled and os.getenv("LOCAL_RANK", None) is None
        self.profiled_actions_enabled = {n: enabled for n in self.PROFILED_FUNCTIONS}
        self.use_cuda = use_cuda
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.sort_by_key = sort_by_key
        self.with_stack = with_stack
        self.group_by_input_shape = group_by_input_shape and record_shapes
        self.use_kineto = use_kineto
        self.use_cpu = use_cpu
        self.emit_nvtx = emit_nvtx
        self.export_to_chrome = export_to_chrome
        self.path_to_export_trace = path_to_export_trace
        if self.sort_by_key not in self.available_sort_by_keys:
            raise MisconfigurationException(
                f"Found sort_by_key: {sort_by_key}. Should be within {self.available_sort_by_keys}. ")

        self.output_fname = output_filename
        self.output_file = None
        if self.output_fname:
            fs = get_filesystem(self.output_fname)
            self.output_file = fs.open(self.output_fname, "w")

        streaming_out = [self.output_file.write] if self.output_file else [log.info]
        super().__init__(output_streams=streaming_out)

    def start(self, action_name: str) -> None:
        if action_name not in self.profiled_actions and action_name in self.PROFILED_FUNCTIONS:
            self.enabled = self.profiled_actions_enabled[action_name]
            self.profiled_actions[action_name] = []
            if self.emit_nvtx:
                self._create_profiler(action_name, torch.cuda.profiler.profile, enter=False)
                self._create_profiler(action_name, torch.autograd.profiler.emit_nvtx)
            else:
                self._create_profiler(action_name, torch.autograd.profiler.profile)

    def _create_profiler(self, action_name, profiler, enter=True):
        init_args = inspect.signature(profiler.__init__).parameters
        profiler_args = {
            k: v for k, v in vars(self).items() if k in init_args
        }
        pr = profiler(**profiler_args)
        if enter:
            pr = pr.__enter__()
        self.profiled_actions[action_name].append(pr)

    def stop(self, action_name: str) -> None:
        if action_name in self.PROFILED_FUNCTIONS and self.enabled:
            profilers = self.profiled_actions.get(action_name)
            if not profilers:
                raise ValueError(  # pragma: no-cover
                    f"Attempting to stop recording an action ({action_name}) which was never started."
                )
            else:
                for pr in profilers[::-1]:
                    self._handle_exit(pr)
        self.profiled_actions_enabled[action_name] = True

    def _handle_exit(self, pr):
        # todo: Find a better solution to exit context manager
        try:
            _ = pr.__exit__(None, None, None)
        except RuntimeError as e:
            if "Expected debug info of type 2" in str(e):
                pass
            elif "can't disable profiler when it's not running" in str(e):
                pass
            elif "generator didn't stop" in str(e):
                pass
            else:
                raise RuntimeError(str(e))

    def summary(self) -> str:
        recorded_stats = {}
        if self.enabled:
            for action_name, pr in self.profiled_actions.items():
                pr = pr[-1]
                if self.export_to_chrome:
                    filename = f"{action_name}_trace.json"
                    path_to_trace = filename if self.path_to_export_trace is None \
                        else os.path.join(self.path_to_export_trace, filename)
                    pr.export_chrome_trace(path_to_trace)
                if self.emit_nvtx:
                    return ""
                else:
                    table = pr.key_averages(
                        group_by_input_shape=self.group_by_input_shape).table(sort_by=self.sort_by_key)
                    recorded_stats[action_name] = table

            # log to standard out
            output_string = f"{os.linesep}Profiler Report{os.linesep}"
            for action, stats in recorded_stats.items():
                output_string += (
                    f"{os.linesep}Profile stats for: {action}{os.linesep}{stats}"
                )

            return output_string
        return ''

    def describe(self):
        """Logs a profile report after the conclusion of the training run."""
        super().describe()
        if self.output_file:
            self.output_file.flush()

    def __del__(self):
        """Close profiler's stream."""
        if self.output_file:
            self.output_file.close()

    @property
    def available_sort_by_keys(self):
        return [
            "cpu_time", "cuda_time", "cpu_time_total",
            "cuda_time_total", "cpu_memory_usage", "cuda_memory_usage",
            "self_cpu_memory_usage", "self_cuda_memory_usage", "count"
        ]
