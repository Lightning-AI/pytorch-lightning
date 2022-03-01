.. _profiler:

#########
Profiling
#########

Profiling your training/testing/inference run can help you identify bottlenecks in your code. The reports can be generated with ``trainer.fit()``,
``trainer.test()``, ``trainer.validate()`` and ``trainer.predict()`` for their respective actions.


------------

****************
Built-in Actions
****************

PyTorch Lightning supports profiling standard actions in the training loop out of the box, including:

- on_train_epoch_start
- on_train_epoch_end
- on_train_batch_start
- model_backward
- on_after_backward
- optimizer_step
- on_train_batch_end
- training_step_end
- on_training_end
- etc...

------------

*******************
Supported Profilers
*******************

Lightning provides the following profilers:

Simple Profiler
===============

If you only wish to profile the standard actions, you can set ``profiler="simple"``. It uses the built-in
:class:`~pytorch_lightning.profiler.simple.SimpleProfiler`.

.. code-block:: python

    # by passing a string
    trainer = Trainer(..., profiler="simple")

    # or by passing an instance
    from pytorch_lightning.profiler import SimpleProfiler

    profiler = SimpleProfiler()
    trainer = Trainer(..., profiler=profiler)

The profiler's results will be printed at the completion of a training ``trainer.fit()``. Find an example of the
:class:`~pytorch_lightning.profiler.simple.SimpleProfiler` report containing a few of the actions:

.. code-block::

    FIT Profiler Report

    -----------------------------------------------------------------------------------------------
    |  Action                                          |  Mean duration (s)	|  Total time (s) |
    -----------------------------------------------------------------------------------------------
    |  run_training_epoch                              |  6.1558         	|  6.1558         |
    |  run_training_batch                              |  0.0022506      	|  0.015754       |
    |  [LightningModule]BoringModel.optimizer_step     |  0.0017477      	|  0.012234       |
    |  [LightningModule]BoringModel.val_dataloader     |  0.00024388     	|  0.00024388     |
    |  on_train_batch_start                            |  0.00014637     	|  0.0010246      |
    |  [LightningModule]BoringModel.teardown           |  2.15e-06       	|  2.15e-06       |
    |  [LightningModule]BoringModel.prepare_data       |  1.955e-06      	|  1.955e-06      |
    |  [LightningModule]BoringModel.on_train_start     |  1.644e-06      	|  1.644e-06      |
    |  [LightningModule]BoringModel.on_train_end       |  1.516e-06      	|  1.516e-06      |
    |  [LightningModule]BoringModel.on_fit_end         |  1.426e-06      	|  1.426e-06      |
    |  [LightningModule]BoringModel.setup              |  1.403e-06      	|  1.403e-06      |
    |  [LightningModule]BoringModel.on_fit_start       |  1.226e-06      	|  1.226e-06      |
    -----------------------------------------------------------------------------------------------

.. note:: Note that there are a lot more actions that will be present in the final report along with percentage and call count for each action.


Advanced Profiler
=================

If you want more information on the functions called during each event, you can use the :class:`~pytorch_lightning.profiler.advanced.AdvancedProfiler`.
This option uses Python's `cProfiler <https://docs.python.org/3/library/profile.html#module-cProfile>`_ to provide an in-depth report of time spent within *each* function called in your code.

.. code-block:: python

    # by passing a string
    trainer = Trainer(..., profiler="advanced")

    # or by passing an instance
    from pytorch_lightning.profiler import AdvancedProfiler

    profiler = AdvancedProfiler()
    trainer = Trainer(..., profiler=profiler)

The profiler's results will be printed at the completion of ``trainer.fit()``. This profiler
report can be quite long, so you can also specify a ``dirpath`` and ``filename`` to save the report instead
of logging it to the output in your terminal. The output below shows the profiling for the action
``get_train_batch``.

.. code-block::

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


PyTorch Profiler
================

Autograd includes a profiler that lets you inspect the cost of different operators
inside your model - both on the CPU and GPU. It uses the built-in :class:`~pytorch_lightning.profiler.pytorch.PyTorchProfiler`.

To read more about the PyTorch Profiler and all its options,
have a look at its `docs <https://pytorch.org/docs/master/profiler.html>`_.

.. code-block:: python

    # by passing a string
    trainer = Trainer(..., profiler="pytorch")

    # or by passing an instance
    from pytorch_lightning.profiler import PyTorchProfiler

    profiler = PyTorchProfiler()
    trainer = Trainer(..., profiler=profiler)


This profiler works with multi-device settings.
If ``filename`` is provided, each rank will save their profiled operation to their own file. The profiler
report can be quite long, so you setting a ``filename`` will save the report instead of logging it to the
output in your terminal. If no filename is given, it will be logged only on rank 0.

The profiler's results will be printed on the completion of ``{fit,validate,test,predict}``.

This profiler will record ``training_step``, ``backward``, ``validation_step``, ``test_step``, and ``predict_step`` by default.
The output below shows the profiling for the action ``training_step``. The user can provide ``PyTorchProfiler(record_functions={...})``
to extend the scope of profiled functions.

.. note::
    When using the PyTorch Profiler, wall clock time will not not be representative of the true wall clock time.
    This is due to forcing profiled operations to be measured synchronously, when many CUDA ops happen asynchronously.
    It is recommended to use this Profiler to find bottlenecks/breakdowns, however for end to end wall clock time use
    the ``SimpleProfiler``.

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

When running with ``PyTorchProfiler(emit_nvtx=True)``, you should run as following:

.. code-block::

    nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

To visualize the profiled operation, you can either:

.. code-block::

    nvvp trace_name.prof

.. code-block::

    python -c 'import torch; print(torch.autograd.profiler.load_nvprof("trace_name.prof"))'


XLA Profiler
============

:class:`~pytorch_lightning.profiler.xla.XLAProfiler` will help you debug and optimize training
workload performance for your models using Cloud TPU performance tools.

.. code-block:: python

    # by passing the `XLAProfiler` alias
    trainer = Trainer(..., profiler="xla")

    # or by passing an instance
    from pytorch_lightning.profiler import XLAProfiler

    profiler = XLAProfiler(port=9001)
    trainer = Trainer(..., profiler=profiler)


Manual Capture via TensorBoard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following instructions are for capturing traces from a running program:

0. This `guide <https://cloud.google.com/tpu/docs/pytorch-xla-performance-profiling-tpu-vm#tpu-vm>`_ will
help you with the Cloud TPU setup with the required installations.

1. Start a `TensorBoard <https://www.tensorflow.org/tensorboard>`_ server. You could view the TensorBoard output at ``http://localhost:9001`` on your local machine, and then open the
``PROFILE`` plugin from the top right dropdown or open ``http://localhost:9001/#profile``

.. code-block:: bash

    tensorboard --logdir ./tensorboard --port 9001

2. Once the code you'd like to profile is running, click on the ``CAPTURE PROFILE`` button. Enter
``localhost:9001`` (default port for XLA Profiler) as the Profile Service URL. Then, enter
the number of milliseconds for the profiling duration, and click ``CAPTURE``

3. Make sure the code is running while you are trying to capture the traces. Also, it would lead to better
performance insights if the profiling duration is longer than the step time.

4. Once the capture is finished, the page will refresh and you can browse through the insights using the
``Tools`` dropdown at the top left


----------------

****************
Custom Profiling
****************

Custom Profiler
===============

You can also configure a custom profiler and pass it to the Trainer. To configure it, subclass :class:`~pytorch_lightning.profiler.base.BaseProfiler`
and override some of its methods. The following is a simple example that profiles the first occurrence and total calls of each action:

.. code-block:: python

    from pytorch_lightning.profiler.base import BaseProfiler
    from collections import defaultdict
    import time


    class ActionCountProfiler(BaseProfiler):
        def __init__(self, dirpath=None, filename=None):
            super().__init__(dirpath=dirpath, filename=filename)
            self._action_count = defaultdict(int)
            self._action_first_occurrence = {}

        def start(self, action_name):
            if action_name not in self._action_first_occurrence:
                self._action_first_occurrence[action_name] = time.strftime("%m/%d/%Y, %H:%M:%S")

        def stop(self, action_name):
            self._action_count[action_name] += 1

        def summary(self):
            res = f"\nProfile Summary: \n"
            max_len = max(len(x) for x in self._action_count)

            for action_name in self._action_count:
                # generate summary for actions called more than once
                if self._action_count[action_name] > 1:
                    res += (
                        f"{action_name:<{max_len}s} \t "
                        + "self._action_first_occurrence[action_name]} \t "
                        + "{self._action_count[action_name]} \n"
                    )

            return res

        def teardown(self, stage):
            self._action_count = {}
            self._action_first_occurrence = {}
            super().teardown(stage=stage)

.. code-block:: python

    trainer = Trainer(..., profiler=ActionCountProfiler())
    trainer.fit(...)


Profile Logic of Your Interest
==============================

You can also reference this profiler in your LightningModule to profile specific actions of interest.
Each profiler has a method ``profile()`` which returns a context handler. Simply pass in the name of
your action that you want to track and the profiler will record performance for code executed within this context.

.. code-block:: python

    from pytorch_lightning.profiler import SimpleProfiler, PassThroughProfiler


    class MyModel(LightningModule):
        def __init__(self, profiler=None):
            self.profiler = profiler or PassThroughProfiler()

        def custom_processing_step(self, data):
            with self.profiler.profile("my_custom_action"):
                ...
            return data


    profiler = SimpleProfiler()
    model = MyModel(profiler)
    trainer = Trainer(profiler=profiler, max_epochs=1)
