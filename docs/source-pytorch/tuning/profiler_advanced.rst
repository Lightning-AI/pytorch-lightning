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
To profile TPU models use the :class:`~lightning.pytorch.profilers.xla.XLAProfiler`

.. code-block:: python

    from lightning.pytorch.profilers import XLAProfiler

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
