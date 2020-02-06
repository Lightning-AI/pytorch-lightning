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
- on_training_end

If you only wish to profile the standard actions, you can set `profiler=True` when constructing
your `Trainer` object.

.. code-block:: python

    trainer = Trainer(..., profiler=True)

The profiler's results will be printed at the completion of a training `fit()`.

.. code-block:: python

    Profiler Report

    Action                  |  Mean duration (s)    |  Total time (s)
    -----------------------------------------------------------------
    on_epoch_start          |  5.993e-06            |  5.993e-06
    get_train_batch         |  0.0087412            |  16.398
    on_batch_start          |  5.0865e-06           |  0.0095372
    model_forward           |  0.0017818            |  3.3408
    model_backward          |  0.0018283            |  3.4282
    on_after_backward       |  4.2862e-06           |  0.0080366
    optimizer_step          |  0.0011072            |  2.0759
    on_batch_end            |  4.5202e-06           |  0.0084753
    on_epoch_end            |  3.919e-06            |  3.919e-06
    on_train_end            |  5.449e-06            |  5.449e-06


If you want more information on the functions called during each event, you can use the `AdvancedProfiler`.
This option uses Python's cProfiler_ to provide a report of time spent on *each* function called within your code.

.. _cProfiler: https://docs.python.org/3/library/profile.html#module-cProfile

.. code-block:: python

    profiler = AdvancedProfiler()
    trainer = Trainer(..., profiler=profiler)

The profiler's results will be printed at the completion of a training `fit()`. This profiler
report can be quite long, so you can also specify an `output_filename` to save the report instead
of logging it to the output in your terminal. The output below shows the profiling for the action
`get_train_batch`.

.. code-block:: python

    Profiler Report

    Profile stats for: get_train_batch
            4869394 function calls (4863767 primitive calls) in 18.893 seconds
    Ordered by: cumulative time
    List reduced from 76 to 10 due to restriction <10>
    ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    3752/1876    0.011    0.000   18.887    0.010 {built-in method builtins.next}
        1876    0.008    0.000   18.877    0.010 dataloader.py:344(__next__)
        1876    0.074    0.000   18.869    0.010 dataloader.py:383(_next_data)
        1875    0.012    0.000   18.721    0.010 fetch.py:42(fetch)
        1875    0.084    0.000   18.290    0.010 fetch.py:44(<listcomp>)
        60000    1.759    0.000   18.206    0.000 mnist.py:80(__getitem__)
        60000    0.267    0.000   13.022    0.000 transforms.py:68(__call__)
        60000    0.182    0.000    7.020    0.000 transforms.py:93(__call__)
        60000    1.651    0.000    6.839    0.000 functional.py:42(to_tensor)
        60000    0.260    0.000    5.734    0.000 transforms.py:167(__call__)

You can also reference this profiler in your LightningModule to profile specific actions of interest.
If you don't want to always have the profiler turned on, you can optionally pass a `PassThroughProfiler`
which will allow you to skip profiling without having to make any code changes. Each profiler has a
method `profile()` which returns a context handler. Simply pass in the name of your action that you want
to track and the profiler will record performance for code executed within this context.

.. code-block:: python

    from pytorch_lightning.utilities.profiler import Profiler, PassThroughProfiler

    class MyModel(LightningModule):
        def __init__(self, hparams, profiler=None):
            self.hparams = hparams
            self.profiler = profiler or PassThroughProfiler()

        def custom_processing_step(self, data):
            with profiler.profile('my_custom_action'):
                # custom processing step
            return data

    profiler = Profiler()
    model = MyModel(hparams, profiler)
    trainer = Trainer(profiler=profiler, max_epochs=1)

"""


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
            sortby = pstats.SortKey.CUMULATIVE
            ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats(sortby)
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
