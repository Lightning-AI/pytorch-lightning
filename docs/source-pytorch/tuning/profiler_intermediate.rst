:orphan:

.. _profiler_intermediate:

############################################
Find bottlenecks in your code (intermediate)
############################################
**Audience**: Users who want to see more granular profiling information

----

**************************
Profile pytorch operations
**************************
To understand the cost of each PyTorch operation, use the :class:`~lightning.pytorch.profilers.pytorch.PyTorchProfiler` built on top of the `PyTorch profiler <https://pytorch.org/docs/master/profiler.html>`__.

.. code-block:: python

    from lightning.pytorch.profilers import PyTorchProfiler

    profiler = PyTorchProfiler()
    trainer = Trainer(profiler=profiler)

The profiler will generate an output like this:

.. code-block::

    Profiler Report

    Profile stats for: training_step
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

.. note::
    When using the PyTorch Profiler, wall clock time will not not be representative of the true wall clock time.
    This is due to forcing profiled operations to be measured synchronously, when many CUDA ops happen asynchronously.
    It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use
    the ``SimpleProfiler``.

----

***************************
Profile a distributed model
***************************
To profile a distributed model, use the :class:`~lightning.pytorch.profilers.pytorch.PyTorchProfiler` with the *filename* argument which will save a report per rank.

.. code-block:: python

    from lightning.pytorch.profilers import PyTorchProfiler

    profiler = PyTorchProfiler(filename="perf-logs")
    trainer = Trainer(profiler=profiler)

With two ranks, it will generate a report like so:

.. code-block::

    Profiler Report: rank 0

    Profile stats for: training_step
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

.. code-block::

    Profiler Report: rank 1

    Profile stats for: training_step
    ---------------------  ---------------  ---------------  ---------------  ---------------  ---------------
    Name                   Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg
    ---------------------  ---------------  ---------------  ---------------  ---------------  ---------------
    t                      42.10%           1.044ms          62.77%           1.055ms          1.055ms
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

This profiler will record ``training_step``, ``validation_step``, ``test_step``, and ``predict_step``.
The output above shows the profiling for the action ``training_step``.

.. note::
    When using the PyTorch Profiler, wall clock time will not not be representative of the true wall clock time.
    This is due to forcing profiled operations to be measured synchronously, when many CUDA ops happen asynchronously.
    It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use
    the ``SimpleProfiler``.

----

*****************************
Visualize profiled operations
*****************************
To visualize the profiled operations, enable **emit_nvtx** in the :class:`~lightning.pytorch.profilers.pytorch.PyTorchProfiler`.

.. code-block:: python

    from lightning.pytorch.profilers import PyTorchProfiler

    profiler = PyTorchProfiler(emit_nvtx=True)
    trainer = Trainer(profiler=profiler)

Then run as following:

.. code-block::

    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

To visualize the profiled operation, you can either use **nvvp**:

.. code-block::

    nvvp trace_name.prof

or python:

.. code-block::

    python -c 'import torch; print(torch.autograd.profiler.load_nvprof("trace_name.prof"))'
