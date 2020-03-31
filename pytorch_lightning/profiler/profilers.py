import cProfile
import io
import os
import pstats
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from contextlib import contextmanager

import numpy as np

from pytorch_lightning import _logger as log


class BaseProfiler(ABC):
    """
    If you wish to write a custom profiler, you should inhereit from this class.
    """

    def __init__(self, output_streams: list = None):
        """
        Params:
            stream_out: callable
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

    def __init__(self, output_filename: str = None):
        """
        Params:
            output_filename (str): optionally save profile results to file instead of printing
                to std out when training is finished.
        """
        self.current_actions = {}
        self.recorded_durations = defaultdict(list)

        self.output_fname = output_filename
        self.output_file = open(self.output_fname, 'w') if self.output_fname else None

        streaming_out = [self.output_file.write] if self.output_file else [log.info]
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

    def summary(self) -> str:
        output_string = "\n\nProfiler Report\n"

        def log_row(action, mean, total):
            return f"{os.linesep}{action:<20s}\t|  {mean:<15}\t|  {total:<15}"

        output_string += log_row("Action", "Mean duration (s)", "Total time (s)")
        output_string += f"{os.linesep}{'-' * 65}"
        for action, durations in self.recorded_durations.items():
            output_string += log_row(
                action, f"{np.mean(durations):.5}", f"{np.sum(durations):.5}",
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

    def __init__(self, output_filename: str = None, line_count_restriction: float = 1.0):
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
        self.output_file = open(self.output_fname, 'w') if self.output_fname else None

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
            output_string += f"{os.linesep}Profile stats for: {action}{os.linesep}{stats}"

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
