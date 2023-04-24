:orphan:

.. _ipu_basic:

Accelerator: IPU training
=========================
**Audience:** Users looking to save money and run large models faster using single or multiple IPU devices.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

----

What is an IPU?
---------------

The Graphcore `Intelligence Processing Unit (IPU) <https://www.graphcore.ai/products/ipu>`__, built for Artificial Intelligence and Machine Learning, consists of many individual cores, called *tiles*, allowing highly parallel computation. Due to the high bandwidth between tiles, IPUs facilitate machine learning loads where parallelization is essential. Because computation is heavily parallelized,

IPUs operate in a different way to conventional accelerators such as CPU/GPUs. IPUs do not require large batch sizes for maximum parallelization, can provide optimizations across the compiled graph and rely on model parallelism to fully utilize tiles for larger models.

IPUs are used to build IPU-PODs, rack-based systems of IPU-Machines for larger workloads. See the `IPU Architecture <https://www.graphcore.ai/products/ipu>`__ for more information.

See the `Graphcore Glossary <https://docs.graphcore.ai/projects/graphcore-glossary/>`__ for the definitions of other IPU-specific terminology.

----

Run on IPU
----------

To enable PyTorch Lightning to utilize the IPU accelerator, simply provide ``accelerator="ipu"`` parameter to the Trainer class.

To use multiple IPUs set the devices to a number that is a power of 2 (i.e: 2, 4, 8, 16, ...)

.. code-block:: python

    # run on as many IPUs as available by default
    trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
    # equivalent to
    trainer = Trainer()

    # run on one IPU
    trainer = Trainer(accelerator="ipu", devices=1)
    # run on multiple IPUs
    trainer = Trainer(accelerator="ipu", devices=8)
    # choose the number of devices automatically
    trainer = Trainer(accelerator="ipu", devices="auto")

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

Please see the `MNIST example <https://github.com/Lightning-AI/lightning/blob/master/examples/pytorch/ipu/mnist_sample.py>`__ which displays most of the limitations and how to overcome them till they are resolved.

* ``self.log`` is not supported in the ``training_step``, ``validation_step``, ``test_step`` or ``predict_step``. This is due to the step function being traced and sent to the IPU devices.
* Since the step functions are traced, branching logic or any form of primitive values are traced into constants. Be mindful as this could lead to errors in your custom code.
* Clipping gradients is not supported.
* It is not possible to use :class:`torch.utils.data.BatchSampler` in your dataloaders if you are using multiple IPUs.
* IPUs handle the data transfer to the device on the host, hence the hooks :meth:`~lightning.pytorch.core.hooks.ModelHooks.transfer_batch_to_device` and
  :meth:`~lightning.pytorch.core.hooks.ModelHooks.on_after_batch_transfer` do not apply here and if you have overridden any of them, an exception will be raised.
