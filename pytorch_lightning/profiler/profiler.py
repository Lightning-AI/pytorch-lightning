from contextlib import contextmanager
from collections import defaultdict
import time
import numpy as np
import cProfile
import pstats
import io
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseProfiler(ABC):
    """
    If you wish to write a custom profiler, you should inhereit from this class.
    """

    @abstractmethod
    def start(self, action_name):
        """
        Defines how to start recording an action.
        """
        pass

    @abstractmethod
    def stop(self, action_name):
        """
        Defines how to record the duration once an action is complete.
        """
        pass

    @contextmanager
    def profile(self, action_name):
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

    def profile_iterable(self, iterable, action_name):
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

    def describe(self):
        """
        Logs a profile report after the conclusion of the training run.
        """
        pass


class PassThroughProfiler(BaseProfiler):
    """
    This class should be used when you don't want the (small) overhead of profiling.
    The Trainer uses this class by default.
    """

    def __init__(self):
        pass

    def start(self, action_name):
        pass

    def stop(self, action_name):
        pass


class Profiler(BaseProfiler):
    """
    This profiler simply records the duration of actions (in seconds) and reports
    the mean duration of each action and the total time spent over the entire training run.
    """

    def __init__(self):
        self.current_actions = {}
        self.recorded_durations = defaultdict(list)

    def start(self, action_name):
        if action_name in self.current_actions:
            raise ValueError(
                f"Attempted to start {action_name} which has already started."
            )
        self.current_actions[action_name] = time.monotonic()

    def stop(self, action_name):
        end_time = time.monotonic()
        if action_name not in self.current_actions:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        start_time = self.current_actions.pop(action_name)
        duration = end_time - start_time
        self.recorded_durations[action_name].append(duration)

    def describe(self):
        output_string = "\n\nProfiler Report\n"

        def log_row(action, mean, total):
            return f"\n{action:<20s}\t|  {mean:<15}\t|  {total:<15}"

        output_string += log_row("Action", "Mean duration (s)", "Total time (s)")
        output_string += f"\n{'-' * 65}"
        for action, durations in self.recorded_durations.items():
            output_string += log_row(
                action, f"{np.mean(durations):.5}", f"{np.sum(durations):.5}",
            )
        output_string += "\n"
        logger.info(output_string)


class AdvancedProfiler(BaseProfiler):
    """
    This profiler uses Python's cProfiler to record more detailed information about
    time spent in each function call recorded during a given action. The output is quite
    verbose and you should only use this if you want very detailed reports.
    """

    def __init__(self, output_filename=None, line_count_restriction=1.0):
        """
        :param output_filename (str): optionally save profile results to file instead of printing
            to std out when training is finished.
        :param line_count_restriction (int|float): this can be used to limit the number of functions
            reported for each action. either an integer (to select a count of lines),
            or a decimal fraction between 0.0 and 1.0 inclusive (to select a percentage of lines)
        """
        self.profiled_actions = {}
        self.output_filename = output_filename
        self.line_count_restriction = line_count_restriction

    def start(self, action_name):
        if action_name not in self.profiled_actions:
            self.profiled_actions[action_name] = cProfile.Profile()
        self.profiled_actions[action_name].enable()

    def stop(self, action_name):
        pr = self.profiled_actions.get(action_name)
        if pr is None:
            raise ValueError(
                f"Attempting to stop recording an action ({action_name}) which was never started."
            )
        pr.disable()

    def describe(self):
        self.recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats('cumulative')
            ps.print_stats(self.line_count_restriction)
            self.recorded_stats[action_name] = s.getvalue()
        if self.output_filename is not None:
            # save to file
            with open(self.output_filename, "w") as f:
                for action, stats in self.recorded_stats.items():
                    f.write(f"Profile stats for: {action}")
                    f.write(stats)
        else:
            # log to standard out
            output_string = "\nProfiler Report\n"
            for action, stats in self.recorded_stats.items():
                output_string += f"\nProfile stats for: {action}\n{stats}"
            logger.info(output_string)
