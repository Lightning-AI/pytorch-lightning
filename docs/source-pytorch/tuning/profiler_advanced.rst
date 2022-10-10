:orphan:

.. _profiler_advanced:

########################################
Find bottlenecks in your code (advanced)
########################################
**Audience**: Users who want to profile their TPU models to find bottlenecks and improve performance.

----

************************
Profile cloud TPU models
************************
To profile TPU models use the :class:`~pytorch_lightning.profilers.xla.XLAProfiler`

.. code-block:: python

    from pytorch_lightning.profilers import XLAProfiler

    profiler = XLAProfiler(port=9001)
    trainer = Trainer(profiler=profiler)

----

*************************************
Capture profiling logs in Tensorboard
*************************************
To capture profile logs in Tensorboard, follow these instructions:

----

0: Setup the required installs
==============================
Use this `guide <https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm#tpu-vm>`_ to help you with the Cloud TPU required installations.

----

1: Start Tensorboard
====================
Start the `TensorBoard <https://www.tensorflow.org/tensorboard>`_ server:

.. code-block:: bash

    tensorboard --logdir ./tensorboard --port 9001

Now open the following url on your browser

.. code-block:: bash

    http://localhost:9001/#profile

----

2: Capture the profile
======================
Once the code you want to profile is running:

1. click on the ``CAPTURE PROFILE`` button.
2. Enter ``localhost:9001`` (default port for XLA Profiler) as the Profile Service URL.
3. Enter the number of milliseconds for the profiling duration
4. Click ``CAPTURE``

----

3: Don't stop your code
=======================
Make sure the code is running while you are trying to capture the traces. It will lead to better performance insights if the profiling duration is longer than the step time.

----

4: View the profiling logs
==========================
Once the capture is finished, the page will refresh and you can browse through the insights using the **Tools** dropdown at the top left



################################################
Find bottlenecks in your code on HPU (advanced)
################################################
**Audience**: Users who want to profile their HPU models to find bottlenecks and improve performance.

----

******************
Profile HPU models
******************
To understand the cost of each PyTorch operation, use the :class:`~pytorch_lightning.profilers.hpu.HPUProfiler` built on top of the `PyTorch profiler <https://pytorch.org/docs/1.12/profiler.html#torch-profiler>`__.

.. code-block:: python

    from pytorch_lightning.profilers import HPUProfiler

    profiler = HPUProfiler()
    trainer = Trainer(profiler=profiler)

The profiler will dump a trace file for each profiler step, ``training_step``, ``backward``, ``validation_step``, ``test_step``, and ``predict_step`` by default.
The user can provide ``PyTorchProfiler(record_functions={...})`` to extend the scope of profiled functions.
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
    Since HPUProfiler extends PyTorch Profiler, when using the HPUProfiler, wall clock time will not be representative of the true wall clock time.
    This is due to forcing profiled operations to be measured synchronously, when many HPU ops happen asynchronously.
    It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use
    the ``SimpleProfiler``.

----

***************************
Profile a distributed model
***************************
To profile a distributed model, use the :class:`~pytorch_lightning.profilers.hpu.HPUProfiler` with the *filename* argument which will save a report per rank.

.. code-block:: python

    from pytorch_lightning.profilers import HPUProfiler

    profiler = HPUProfiler(filename="perf-logs")
    trainer = Trainer(profiler=profiler)

----

*****************************
Visualize profiled operations
*****************************
To visualize the profiled operations, enable **export_to_chrome** in the :class:`~pytorch_lightning.profilers.hpu.HPUProfiler` (Default: True).

.. code-block:: python

    from pytorch_lightning.profilers import HPUProfiler

    profiler = HPUProfiler(export_to_chrome=True)
    trainer = Trainer(profiler=profiler)

Then run the model. Once profiler is finished, load the trace either in tensorboard or chrome browser:

.. code-block::

    tensorboard --logdir <path to trace>

Or load it in chrome tracer:

.. code-block::

    chrome://tracing


----

************************************
Using Simple and Advanced Profilers
************************************

Simple and advanced profilers are compatible with HPU. Please refer to `Find Bottlenecks in your code (Basic) <https://pytorch-lightning.readthedocs.io/en/stable/tuning/profiler_basic.html>`__ for more information on how to use them.
Note that these profilers will not profile the HPU activity. Please use HPUProfiler instead.
