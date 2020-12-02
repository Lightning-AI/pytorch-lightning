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

If you only wish to profile the standard actions, you can set `profiler="simple"`
when constructing your `Trainer` object.

.. code-block:: python

    trainer = Trainer(..., profiler="simple")

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


Autograd Profiling
------------------
If you would like to focus your profiling on the PyTorch-specific components, you should use the autograd
profiler. This leverages the native `torch.autograd.profiler`_ context manager to accurately measure time
spent on PyTorch ops on both the CPU and GPU.

.. _`torch.autograd.profiler`: https://pytorch.org/docs/stable/autograd.html#profiler

.. code-block:: python

    profiler = AutogradProfiler()
    trainer = Trainer(..., profiler=profiler)

The profiler's results will be printed at the completion of a training `fit()`. This profiler
report can be quite long, so you can also specify an `output_filename` to save the report instead
of logging it to the output in your terminal. The output below shows the profiling for the action
`model_forward`.

.. code-block:: python

    Profiler Report

    Profile stats for: model_forward
    --------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
    Name                        Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg     Number of Calls
    --------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
    batch_norm                  0.05%            4.000us          65.88%           5.365ms          5.365ms          1
    _batch_norm_impl_index      0.06%            5.000us          65.83%           5.361ms          5.361ms          1
    native_batch_norm           65.75%           5.355ms          65.75%           5.355ms          5.355ms          1
    addmm                       13.03%           1.061ms          13.03%           1.061ms          530.500us        2
    dropout                     0.10%            8.000us          8.53%            695.000us        695.000us        1
    mul                         6.47%            527.000us        6.47%            527.000us        175.667us        3
    tanh                        4.44%            362.000us        4.44%            362.000us        362.000us        1
    unsigned short              3.76%            306.000us        3.76%            306.000us        153.000us        2
    log_softmax                 0.04%            3.000us          3.25%            265.000us        265.000us        1
    _log_softmax                3.22%            262.000us        3.22%            262.000us        262.000us        1
    bernoulli_                  1.71%            139.000us        1.71%            139.000us        139.000us        1
    div_                        0.52%            42.000us         0.52%            42.000us         42.000us         1
    nll_loss                    0.04%            3.000us          0.33%            27.000us         27.000us         1
    view                        0.32%            26.000us         0.32%            26.000us         26.000us         1
    nll_loss_forward            0.29%            24.000us         0.29%            24.000us         24.000us         1
    add                         0.11%            9.000us          0.11%            9.000us          9.000us          1
    empty                       0.05%            4.000us          0.05%            4.000us          2.000us          2
    empty_like                  0.01%            1.000us          0.05%            4.000us          4.000us          1
    detach                      0.04%            3.000us          0.04%            3.000us          1.000us          3
    --------------------------  ---------------  ---------------  ---------------  ---------------  ---------------  ---------------
    Self CPU time total: 8.144ms



Advanced Profiling
------------------

If you want more information on the functions called during each event, you can use the `AdvancedProfiler`.
This option uses Python's cProfiler_ to provide a report of time spent on *each* function called within your code.

.. _cProfiler: https://docs.python.org/3/library/profile.html#module-cProfile

.. code-block:: python

    trainer = Trainer(..., profiler="advanced")

    or

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



Profiling custom events
-----------------------

You can also reference this profiler in your LightningModule to profile specific actions of interest.
If you don't want to always have the profiler turned on, you can optionally pass a `PassThroughProfiler`
which will allow you to skip profiling without having to make any code changes. Each profiler has a
method `profile()` which returns a context handler. Simply pass in the name of your action that you want
to track and the profiler will record performance for code executed within this context.

.. code-block:: python

    from pytorch_lightning.profiler import Profiler, PassThroughProfiler

    class MyModel(LightningModule):
        def __init__(self, profiler=None):
            self.profiler = profiler or PassThroughProfiler()

        def custom_processing_step(self, data):
            with profiler.profile('my_custom_action'):
                # custom processing step
            return data

    profiler = Profiler()
    model = MyModel(profiler)
    trainer = Trainer(profiler=profiler, max_epochs=1)

"""

from pytorch_lightning.profiler.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    AutogradProfiler,
    PassThroughProfiler,
    BaseProfiler,
)

__all__ = [
    'BaseProfiler',
    'SimpleProfiler',
    'AdvancedProfiler',
    'AutogradProfiler',
    'PassThroughProfiler',
]
