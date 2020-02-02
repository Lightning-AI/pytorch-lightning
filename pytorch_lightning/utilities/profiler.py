"""
Profiling your training run can help you understand if there are any bottlenecks in your code.

PyTorch Lightning supports profiling standard actions in the training loop out of the box, including:
- enumerate  # TODO
- exaples  # TODO
- here  # TODO

"""


from contextlib import contextmanager
from collections import defaultdict
import time
import numpy as np
import cProfile
import pstats
import io
from abc import ABC, abstractmethod


class BaseProfiler(ABC):
    @abstractmethod
    def start(self, action_name):
        """
        defines how to start recording an action
        """
        pass

    @abstractmethod
    def stop(self, action_name):
        """
        defines how to record the duration once an action is complete
        """
        pass

    @contextmanager
    def profile(self, action_name):
        """
        yields a context manager to encapsulate the scope of a profiled action

        with self.profile('load training data'):
            # load training data code

        the profiler will start once you've entered the context and automatically stop
        once you exit the code block
        """
        self.start(action_name)
        yield action_name
        self.stop(action_name)

    def describe(self):
        """
        prints a report after the conclusion of the profiled training run
        """
        pass


class Profiler(BaseProfiler):
    """
    this profiler simply records the duration of actions (in seconds) and reports
    the mean and standard deviation of each action duration over the entire training run
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
        def print_row(action, mean, std_dev):
            print(f"{action}\t|\t{mean:.4}\t|\t{std_dev:.4}")

        print_row("Action", "Mean duration", "Std deviation")
        print("-" * 40)
        for action, durations in self.recorded_durations.items():
            print_row(action, np.mean(durations), np.std(durations))


class AdvancedProfiler(BaseProfiler):
    """
    this profiler uses Python's cProfiler to record more detailed information about
    time spent in each function call recorded during a given action
    """
    def __init__(self):
        self.profiled_actions = {}

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

    def describe(self, line_count_restriction=1.0):
        self.recorded_stats = {}
        for action_name, pr in self.profiled_actions.items():
            s = io.StringIO()
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
            ps.print_stats(line_count_restriction)
            self.recorded_stats[action_name] = s.getvalue()
        for action, stats in self.recorded_stats.items():
            print(f"Profile stats for: {action}")
            print(stats)


if __name__ == '__main__.py':

    p = Profiler()

    with p.profile("test"):
        time.sleep(5)

    with p.profile("test"):
        time.sleep(2)

    with p.profile("test"):
        time.sleep(4)

    with p.profile("ok"):
        time.sleep(1)

    p.describe()

    ap = AdvancedProfiler()

    with ap.profile("test"):
        time.sleep(5)

    with ap.profile("test"):
        time.sleep(2)

    with ap.profile("test"):
        time.sleep(4)

    with ap.profile("ok"):
        time.sleep(1)

    ap.describe()
