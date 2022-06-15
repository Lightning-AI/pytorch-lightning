:orphan:

.. _ipu_basic:

Accelerator: IPU training
=========================
**Audience:** Users looking to save money and run large models faster using single or multiple IPU devices.

----

What is an IPU?
---------------

The Graphcore `Intelligence Processing Unit (IPU) <https://www.graphcore.ai/products/ipu>`__, built for Artificial Intelligence and Machine Learning, consists of many individual cores, called *tiles*, allowing highly parallel computation. Due to the high bandwidth between tiles, IPUs facilitate machine learning loads where parallelization is essential. Because computation is heavily parallelized,

IPUs operate in a different way to conventional accelerators such as CPU/GPUs. IPUs do not require large batch sizes for maximum parallelization, can provide optimizations across the compiled graph and rely on model parallelism to fully utilize tiles for larger models.

IPUs are used to build IPU-PODs, rack-based systems of IPU-Machines for larger workloads. See the `IPU Architecture <https://www.graphcore.ai/products/ipu>`__ for more information.

See the `Graphcore Glossary <https://docs.graphcore.ai/projects/graphcore-glossary/>`__ for the definitions of other IPU-specific terminology.

.. note::
  IPU support is experimental and a work in progress (see :ref:`known-limitations`). If you run into any problems, please leave an issue.

----

Run on 1 IPU
------------
To use a single IPU, set the accelerator and devices argument.

.. code-block:: python

    trainer = pl.Trainer(accelerator="ipu", devices=1)

----

Run on multiple IPUs
--------------------
To use multiple IPUs set the devices to a number that is a power of 2 (i.e: 2, 4, 8, 16, ...)

.. code-block:: python

    trainer = pl.Trainer(accelerator="ipu", devices=8)

----

How to access IPUs
------------------

To use IPUs you must have access to a system with IPU devices. To get access see `get started <https://www.graphcore.ai/getstarted>`__.

You must ensure that the IPU system has enabled the PopART and Poplar packages from the SDK. Instructions are in the Get Started guide for your IPU system, on the Graphcore `documents portal <https://docs.graphcore.ai/page/getting-started.html>`__.

----

.. _known-limitations:

Known limitations
-----------------

Currently there are some known limitations that are being addressed in the near future to make the experience seamless when moving from different devices.

Please see the `MNIST example <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/examples/pl_ipu/mnist_sample.py>`__ which displays most of the limitations and how to overcome them till they are resolved.

* ``self.log`` is not supported in the ``training_step``, ``validation_step``, ``test_step`` or ``predict_step``. This is due to the step function being traced and sent to the IPU devices. We're actively working on fixing this
* Multiple optimizers are not supported. ``training_step`` only supports returning one loss from the ``training_step`` function as a result
* Since the step functions are traced, branching logic or any form of primitive values are traced into constants. Be mindful as this could lead to errors in your custom code
* Clipping gradients is not supported
