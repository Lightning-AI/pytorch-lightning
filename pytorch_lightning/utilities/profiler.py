"""
Profiling your training run can help you understand if there are any bottlenecks in your code.

PyTorch Lightning supports profiling standard actions in the training loop out of the box, including:
- on_epoch_start
- on_epoch_end
- on_batch_start
- tbptt_split_batch
- model_forward
- model_backward
- on_after_backward
- optimizer_step
- on_batch_end
- training_end

If you only wish to profile the standard actions, you can construct a Profiler object and simply
pass it into the Trainer.

.. code-block:: python
    profiler = Profiler()
    trainer = Trainer(..., profiler=profiler)

You can also reference this profiler to profiler any arbitrary code.

.. code-block:: python
    with profiler.profile('my_custom_action'):
        my_custom_action()
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


class PassThroughProfiler(BaseProfiler):
    """
    this can be used when you don't want to profile your runs
    """

    def __init__(self):
        pass

    def start(self):
        pass

    def stop(self):
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
            print(f"{action:<20s}\t|  {mean:<15}\t|  {std_dev:<15}")

        print_row("Action", "Mean duration (s)", "Std dev.")
        print("-" * 60)
        for action, durations in self.recorded_durations.items():
            print_row(action, f"{np.mean(durations):.5}", f"{np.std(durations):.5}")


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


if __name__ == "__main__.py":

    p = Profiler()

    with p.profile("context handler"):
        time.sleep(5)
        a = np.random.randn(3000, 2)
        b = a + 2
        c = b / 3

    with p.profile("context handler"):
        time.sleep(1)
        a = np.random.randn(3000, 2)
        b = a + 2
        c = b / 3

    p.start("manual")
    time.sleep(5)
    a = np.random.randn(3000, 2)
    b = a + 2
    c = b / 3
    p.stop("manual")

    p.describe()

    ap = AdvancedProfiler()

    with ap.profile("context handler"):
        time.sleep(5)
        a = np.random.randn(3000, 2)
        b = a + 2
        c = b / 3

    with ap.profile("context handler"):
        time.sleep(1)
        a = np.random.randn(3000, 2)
        b = a + 2
        c = b / 3

    ap.start("manual")
    time.sleep(5)
    a = np.random.randn(3000, 2)
    b = a + 2
    c = b / 3
    ap.stop("manual")

    ap.describe()
