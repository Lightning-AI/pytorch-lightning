"""
Profiling your training run can help you understand if there are any bottlenecks in your code.


Built-in checks
---------------

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
- training_step_end
- on_training_end

Enable simple profiling
-----------------------

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


Advanced Profiling
--------------------

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
        1876     0.008    0.000   18.877    0.010 dataloader.py:344(__next__)
        1876     0.074    0.000   18.869    0.010 dataloader.py:383(_next_data)
        1875     0.012    0.000   18.721    0.010 fetch.py:42(fetch)
        1875     0.084    0.000   18.290    0.010 fetch.py:44(<listcomp>)
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

    from pytorch_lightning.profiler import Profiler, PassThroughProfiler

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

from pytorch_lightning.profiler.profilers import SimpleProfiler, AdvancedProfiler, PassThroughProfiler, BaseProfiler

__all__ = [
    'BaseProfiler',
    'SimpleProfiler',
    'AdvancedProfiler',
    'PassThroughProfiler',
]
