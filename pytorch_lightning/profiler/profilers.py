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
import io
import logging
import os
import pstats
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TextIO, Tuple, Union

import numpy as np

from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.cloud_io import get_filesystem

log = logging.getLogger(__name__)


class AbstractProfiler(ABC):
    """Specification of a profiler."""

    @abstractmethod
    def start(self, action_name: str) -> None:
        """Defines how to start recording an action."""

    @abstractmethod
    def stop(self, action_name: str) -> None:
        """Defines how to record the duration once an action is complete."""

    @abstractmethod
    def summary(self) -> str:
        """Create profiler summary in text format."""

    @abstractmethod
    def setup(self, **kwargs: Any) -> None:
        """Execute arbitrary pre-profiling set-up steps as defined by subclass."""

    @abstractmethod
    def teardown(self, **kwargs: Any) -> None:
        """Execute arbitrary post-profiling tear-down steps as defined by subclass."""


class BaseProfiler(AbstractProfiler):
    """
    If you wish to write a custom profiler, you should inherit from this class.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> None:
        self.dirpath = dirpath
        self.filename = filename
        if output_filename is not None:
            rank_zero_warn(
                "`Profiler` signature has changed in v1.3. The `output_filename` parameter has been removed in"
                " favor of `dirpath` and `filename`. Support for the old signature will be removed in v1.5",
                DeprecationWarning
            )
            filepath = Path(output_filename)
            self.dirpath = filepath.parent
            self.filename = filepath.stem

        self._output_file: Optional[TextIO] = None
        self._write_stream: Optional[Callable] = None
        self._local_rank: Optional[int] = None
        self._log_dir: Optional[str] = None
        self._stage: Optional[str] = None

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

    def _rank_zero_info(self, *args, **kwargs) -> None:
        if self._local_rank in (None, 0):
            log.info(*args, **kwargs)

    def _prepare_filename(self, extension: str = ".txt") -> str:
        filename = ""
        if self._stage is not None:
            filename += f"{self._stage}-"
        filename += str(self.filename)
        if self._local_rank is not None:
            filename += f"-{self._local_rank}"
        filename += extension
        return filename

    def _prepare_streams(self) -> None:
        if self._write_stream is not None:
            return
        if self.filename:
            filepath = os.path.join(self.dirpath, self._prepare_filename())
            fs = get_filesystem(filepath)
            file = fs.open(filepath, "a")
            self._output_file = file
            self._write_stream = file.write
        else:
            self._write_stream = self._rank_zero_info

    def describe(self) -> None:
        """Logs a profile report after the conclusion of run."""
        # there are pickling issues with open file handles in Python 3.6
        # so to avoid them, we open and close the files within this function
        # by calling `_prepare_streams` and `teardown`
        self._prepare_streams()
        summary = self.summary()
        if summary:
            self._write_stream(summary)
        if self._output_file is not None:
            self._output_file.flush()
        self.teardown(stage=self._stage)

    def _stats_to_str(self, stats: Dict[str, str]) -> str:
        stage = f"{self._stage.upper()} " if self._stage is not None else ""
        output = [stage + "Profiler Report"]
        for action, value in stats.items():
            header = f"Profile stats for: {action}"
            if self._local_rank is not None:
                header += f" rank: {self._local_rank}"
            output.append(header)
            output.append(value)
        return os.linesep.join(output)

    def setup(
        self,
        stage: Optional[str] = None,
        local_rank: Optional[int] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        """Execute arbitrary pre-profiling set-up steps."""
        self._stage = stage
        self._local_rank = local_rank
        self._log_dir = log_dir
        self.dirpath = self.dirpath or log_dir

    def teardown(self, stage: Optional[str] = None) -> None:
        """
        Execute arbitrary post-profiling tear-down steps.

        Closes the currently open file and stream.
        """
        self._write_stream = None
        if self._output_file is not None:
            self._output_file.close()
            self._output_file = None  # can't pickle TextIOWrapper

    def __del__(self) -> None:
        self.teardown(stage=self._stage)

    def start(self, action_name: str) -> None:
        raise NotImplementedError

    def stop(self, action_name: str) -> None:
        raise NotImplementedError

    def summary(self) -> str:
        raise NotImplementedError

    @property
    def local_rank(self) -> int:
        return 0 if self._local_rank is None else self._local_rank


class PassThroughProfiler(BaseProfiler):
    """
    This class should be used when you don't want the (small) overhead of profiling.
    The Trainer uses this class by default.
    """

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

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        extended: bool = True,
        output_filename: Optional[str] = None,
    ) -> None:
        """
        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

        Raises:
            ValueError:
                If you attempt to start an action which has already started, or
                if you attempt to stop recording an action which was never started.
        """
        super().__init__(dirpath=dirpath, filename=filename, output_filename=output_filename)
        self.current_actions: Dict[str, float] = {}
        self.recorded_durations = defaultdict(list)
        self.extended = extended
        self.start_time = time.monotonic()

    def start(self, action_name: str) -> None:
        if action_name in self.current_actions:
            raise ValueError(f"Attempted to start {action_name} which has already started.")
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name: str) -> None:
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(f"Attempting to stop recording an action ({action_name}) which was never started.")
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def _make_report(self) -> Tuple[list, float]:
        total_duration = time.monotonic() - self.start_time
        report = [[a, d, 100. * np.sum(d) / total_duration] for a, d in self.recorded_durations.items()]
        report.sort(key=lambda x: x[2], reverse=True)
        return report, total_duration

    def summary(self) -> str:
        sep = os.linesep
        output_string = ""
        if self._stage is not None:
            output_string += f"{self._stage.upper()} "
        output_string += f"Profiler Report{sep}"

        if self.extended:

            if len(self.recorded_durations) > 0:
                max_key = np.max([len(k) for k in self.recorded_durations.keys()])

                def log_row(action, mean, num_calls, total, per):
                    row = f"{sep}{action:<{max_key}s}\t|  {mean:<15}\t|"
                    row += f"{num_calls:<15}\t|  {total:<15}\t|  {per:<15}\t|"
                    return row

                output_string += log_row("Action", "Mean duration (s)", "Num calls", "Total time (s)", "Percentage %")
                output_string_len = len(output_string)
                output_string += f"{sep}{'-' * output_string_len}"
                report, total_duration = self._make_report()
                output_string += log_row("Total", "-", "_", f"{total_duration:.5}", "100 %")
                output_string += f"{sep}{'-' * output_string_len}"
                for action, durations, duration_per in report:
                    output_string += log_row(
                        action,
                        f"{np.mean(durations):.5}",
                        f"{len(durations):}",
                        f"{np.sum(durations):.5}",
                        f"{duration_per:.5}",
                    )
        else:

            def log_row(action, mean, total):
                return f"{sep}{action:<20s}\t|  {mean:<15}\t|  {total:<15}"

            output_string += log_row("Action", "Mean duration (s)", "Total time (s)")
            output_string += f"{sep}{'-' * 65}"

            for action, durations in self.recorded_durations.items():
                output_string += log_row(action, f"{np.mean(durations):.5}", f"{np.sum(durations):.5}")
        output_string += sep
        return output_string


class AdvancedProfiler(BaseProfiler):
    """
    This profiler uses Python's cProfiler to record more detailed information about
    time spent in each function call recorded during a given action. The output is quite
    verbose and you should only use this if you want very detailed reports.
    """

    def __init__(
        self,
        dirpath: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        line_count_restriction: float = 1.0,
        output_filename: Optional[str] = None,
    ) -> None:
        """
        Args:
            dirpath: Directory path for the ``filename``. If ``dirpath`` is ``None`` but ``filename`` is present, the
                ``trainer.log_dir`` (from :class:`~pytorch_lightning.loggers.tensorboard.TensorBoardLogger`)
                will be used.

            filename: If present, filename where the profiler results will be saved instead of printing to stdout.
                The ``.txt`` extension will be used automatically.

            line_count_restriction: this can be used to limit the number of functions
                reported for each action. either an integer (to select a count of lines),
                or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)

        Raises:
            ValueError:
                If you attempt to stop recording an action which was never started.
        """
        super().__init__(dirpath=dirpath, filename=filename, output_filename=output_filename)
        self.profiled_actions: Dict[str, cProfile.Profile] = {}
        self.line_count_restriction = line_count_restriction

    def start(self, action_name: str) -> None:
        if action_name not in self.profiled_actions:
            self.profiled_actions[action_name] = cProfile.Profile()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name: str) -> None:
        pr = self.profiled_actions.get(action_name)
        if pr is None:
            raise ValueError(f"Attempting to stop recording an action ({action_name}) which was never started.")
        pr.disable()

    def summary(self) -> str:
        recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
            ps.print_stats(self.line_count_restriction)
            recorded_stats[action_name] = s.getvalue()
        return self._stats_to_str(recorded_stats)

    def teardown(self, stage: Optional[str] = None) -> None:
        super().teardown(stage=stage)
        self.profiled_actions = {}

    def __reduce__(self):
        # avoids `TypeError: cannot pickle 'cProfile.Profile' object`
        return (
            self.__class__,
            tuple(),
            dict(dirpath=self.dirpath, filename=self.filename, line_count_restriction=self.line_count_restriction),
        )
