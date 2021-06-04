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


PyTorch Profiling
-----------------

Autograd includes a profiler that lets you inspect the cost of different operators
inside your model - both on the CPU and GPU.

To read more about the PyTorch Profiler and all its options,
have a look at its `docs <https://pytorch.org/docs/master/profiler.html>`__

.. code-block:: python

    trainer = Trainer(..., profiler="pytorch")

    or

    profiler = PyTorchProfiler(...)
    trainer = Trainer(..., profiler=profiler)


This profiler works with PyTorch ``DistributedDataParallel``.
If ``filename`` is provided, each rank will save their profiled operation to their own file. The profiler
report can be quite long, so you setting a ``filename`` will save the report instead of logging it to the
output in your terminal. If no filename is given, it will be logged only on rank 0.

The profiler's results will be printed on the completion of ``{fit,validate,test,predict}``.

This profiler will record ``training_step_and_backward``, ``training_step``, ``backward``,
``validation_step``, ``test_step``, and ``predict_step`` by default.
The output below shows the profiling for the action ``training_step_and_backward``.
The user can provide ``PyTorchProfiler(record_functions={...})`` to extend the scope of profiled functions.

.. note:: When using the PyTorch Profiler, wall clock time will not not be representative of the true wall clock time. This is due to forcing profiled operations to be measured synchronously, when many CUDA ops happen asynchronously. It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use the `SimpleProfiler`.   # noqa E501

.. code-block:: python

    Profiler Report

    Profile stats for: training_step_and_backward
    ---------------------  ---------------  ---------------  ---------------  ---------------  ---------------
    Name                   Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg
    ---------------------  ---------------  ---------------  ---------------  ---------------  ---------------
    t                      62.10%           1.044ms          62.77%           1.055ms          1.055ms
    addmm                  32.32%           543.135us        32.69%           549.362us        549.362us
    mse_loss               1.35%            22.657us         3.58%            60.105us         60.105us
    mean                   0.22%            3.694us          2.05%            34.523us         34.523us
    div_                   0.64%            10.756us         1.90%            32.001us         16.000us
    ones_like              0.21%            3.461us          0.81%            13.669us         13.669us
    sum_out                0.45%            7.638us          0.74%            12.432us         12.432us
    transpose              0.23%            3.786us          0.68%            11.393us         11.393us
    as_strided             0.60%            10.060us         0.60%            10.060us         3.353us
    to                     0.18%            3.059us          0.44%            7.464us          7.464us
    empty_like             0.14%            2.387us          0.41%            6.859us          6.859us
    empty_strided          0.38%            6.351us          0.38%            6.351us          3.175us
    fill_                  0.28%            4.782us          0.33%            5.566us          2.783us
    expand                 0.20%            3.336us          0.28%            4.743us          4.743us
    empty                  0.27%            4.456us          0.27%            4.456us          2.228us
    copy_                  0.15%            2.526us          0.15%            2.526us          2.526us
    broadcast_tensors      0.15%            2.492us          0.15%            2.492us          2.492us
    size                   0.06%            0.967us          0.06%            0.967us          0.484us
    is_complex             0.06%            0.961us          0.06%            0.961us          0.481us
    stride                 0.03%            0.517us          0.03%            0.517us          0.517us
    ---------------------  ---------------  ---------------  ---------------  ---------------  ---------------
    Self CPU time total: 1.681ms

When running with `PyTorchProfiler(emit_nvtx=True)`. You should run as following::

    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

To visualize the profiled operation, you can either:

Use::

    nvvp trace_name.prof

Or::

    python -c 'import torch; print(torch.autograd.profiler.load_nvprof("trace_name.prof"))'

"""
from pytorch_lightning.profiler.advanced import AdvancedProfiler
from pytorch_lightning.profiler.base import AbstractProfiler, BaseProfiler, PassThroughProfiler
from pytorch_lightning.profiler.pytorch import PyTorchProfiler
from pytorch_lightning.profiler.simple import SimpleProfiler

__all__ = [
    'AbstractProfiler',
    'BaseProfiler',
    'AdvancedProfiler',
    'PassThroughProfiler',
    'PyTorchProfiler',
    'SimpleProfiler',
]
