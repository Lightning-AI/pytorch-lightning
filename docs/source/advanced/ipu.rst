.. _ipu:

IPU support
===========

.. note::
    IPU Support is experimental and a work in progress (see :ref:`Known Limitations`). If you run into an problems, please leave an issue.

Lightning supports `GraphCores' Information Processing Units (IPUs) <https://www.graphcore.ai/products/ipu>`_, processors built for Artificial Intelligence and Machine Learning.

IPU Terminology
---------------

TODO

How to access IPUs
------------------

To use IPUs you must have access to a server with IPU devices attached. To get access see `getting started <https://www.graphcore.ai/getstarted>`_.

Training with IPUs
------------------

Specify the number of IPUs to train with. Note that when training with IPUs, you must select 1 or a power of 2 number of IPUs (i.e 2/4/8..).

.. code-block:: python

    trainer = pl.Trainer(ipus=8) # Train using data parallel on 8 IPUs

IPUs only support specifying a single number to allocate devices, which is handled via the underlying libraries.

Mixed Precision & 16 bit precision
----------------------------------

Lightning also supports training in mixed precision with IPUs.
By default, IPU training will use 32-bit precision. To enable mixed precision,
set the precision flag.

.. note::
    Currently there is no dynamic scaling of the loss with mixed precision training.

.. code-block:: python

    import pytorch_lightning as pl

    my_model = MyLightningModule()
    trainer = pl.Trainer(ipus=8, precision=16)
    trainer.fit(my_model)

You can also use pure 16-bit training, where the weights are also in 16 bit precision.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.plugins import IPUPlugin

    my_model = MyLightningModule()
    trainer = pl.Trainer(ipus=8, precision=16, plugins=IPUPlugin(convert_model_to_half=True))
    trainer.fit(my_model)

Advanced IPU Options
--------------------

IPUs provide further optimizations to speed up training. By using the ``IPUPlugin`` we can set the ``device_iterations``, which controls the number of iterations run direcly on the IPU devices before returning to host.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.plugins import IPUPlugin

    my_model = MyLightningModule()
    trainer = pl.Trainer(ipus=8, plugins=IPUPlugin(device_iterations=32))
    trainer.fit(my_model)


You can also override all options by passing the ``poptorch.Options`` to the plugin. See `poptorch options documentation <https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html>`_ for more information.

.. code-block:: python

    import pytorch_lightning as pl
    from pytorch_lightning.plugins import IPUPlugin

    my_model = MyLightningModule()
    inference_opts = poptorch.Options()
    inference_opts.deviceIterations(32)

    training_opts = poptorch.Options()
    training_opts.deviceIterations(32)

    trainer = Trainer(
        ipus=8,
        plugins=IPUPlugin(inference_opts=inference_opts, training_opts=training_opts)
    )
    trainer.fit(my_model)


IPU Profiler
------------

Model Pipe Parallelism
----------------------

TODO

Known Limitations
-----------------

Currently there are some known limitations that are being addressed in the near future to make the experience seamless when moving from different devices.

Please see the `MNIST example <https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/ipu_examples/mnist.py>`__ which displays most of the limitations and how to overcome them till they are resolved.

* ``self.log`` is not supported in the ``training_step``, ``validation_step``, ``test_step`` or ``predict_step``. This is due to the step function being traced and sent to the IPU devices. We're actively working on fixing this
* ``training_step`` only supports returning one loss from the ``training_step`` function
* Since the step functions are traced, branching logic or any form of primitive values are traced into constants. Be mindful as this could lead to errors in your custom code
* Multiple optimizers are not supported
