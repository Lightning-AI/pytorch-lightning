:orphan:

.. _ipu_advanced:

Accelerator: IPU training
=========================
**Audience:** Users looking to customize IPU training for massive models.

.. warning::  This is an :ref:`experimental <versioning:Experimental API>` feature.

----

Advanced IPU options
--------------------

IPUs provide further optimizations to speed up training. By using the ``IPUStrategy`` we can set the ``device_iterations``, which controls the number of iterations run directly on the IPU devices before returning to the host. Increasing the number of on-device iterations will improve throughput, as there is less device to host communication required.

.. note::

    When using model parallelism, it is a hard requirement to increase the number of device iterations to ensure we fully saturate the devices via micro-batching. see :ref:`ipu-model-parallelism` for more information.

.. code-block:: python

    import lightning.pytorch as pl
    from lightning.pytorch.strategies import IPUStrategy

    model = MyLightningModule()
    trainer = pl.Trainer(accelerator="ipu", devices=8, strategy=IPUStrategy(device_iterations=32))
    trainer.fit(model)

Note that by default we return the last device iteration loss. You can override this by passing in your own ``poptorch.Options`` and setting the AnchorMode as described in the `PopTorch documentation <https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/reference.html#poptorch.Options.anchorMode>`__.

.. code-block:: python

    import poptorch
    import lightning.pytorch as pl
    from lightning.pytorch.strategies import IPUStrategy

    model = MyLightningModule()
    inference_opts = poptorch.Options()
    inference_opts.deviceIterations(32)

    training_opts = poptorch.Options()
    training_opts.anchorMode(poptorch.AnchorMode.All)
    training_opts.deviceIterations(32)

    trainer = Trainer(
        accelerator="ipu", devices=8, strategy=IPUStrategy(inference_opts=inference_opts, training_opts=training_opts)
    )
    trainer.fit(model)

You can also override all options by passing the ``poptorch.Options`` to the plugin. See `PopTorch options documentation <https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/batching.html>`__ for more information.

----

.. _ipu-model-parallelism:

Model parallelism
-----------------

Due to the IPU architecture, larger models should be parallelized across IPUs by design. Currently PopTorch provides the capabilities via annotations as described in `parallel execution strategies <https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/overview.html#execution-strategies>`__.

Below is an example using the block annotation in a LightningModule.

.. note::

    Currently, when using model parallelism we do not infer the number of IPUs required for you. This is done via the annotations themselves. If you specify 4 different IDs when defining Blocks, this means your model will be split onto 4 different IPUs.

    This is also mutually exclusive with the Trainer flag. In other words, if your model is split onto 2 IPUs and you set ``Trainer(accelerator="ipu", devices=4)`` this will require 8 IPUs in total: data parallelism will be used to replicate the two-IPU model 4 times.

    When pipelining the model you must also increase the `device_iterations` to ensure full data saturation of the devices data, i.e whilst one device in the model pipeline processes a batch of data, the other device can start on the next batch. For example if the model is split onto 4 IPUs, we require `device_iterations` to be at-least 4.


.. code-block:: python

    import lightning.pytorch as pl
    import poptorch


    class MyLightningModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            # This will place layer1, layer2+layer3, layer4, softmax on different IPUs at runtime.
            # BeginBlock will start a new id for all layers within this block
            self.layer1 = poptorch.BeginBlock(torch.nn.Linear(5, 10), ipu_id=0)

            # This layer starts a new block,
            # adding subsequent layers to this current block at runtime
            # till the next block has been declared
            self.layer2 = poptorch.BeginBlock(torch.nn.Linear(10, 5), ipu_id=1)
            self.layer3 = torch.nn.Linear(5, 5)

            # Create new blocks
            self.layer4 = poptorch.BeginBlock(torch.nn.Linear(5, 5), ipu_id=2)
            self.softmax = poptorch.BeginBlock(torch.nn.Softmax(dim=1), ipu_id=3)

        ...


    model = MyLightningModule()
    trainer = pl.Trainer(accelerator="ipu", devices=8, strategy=IPUStrategy(device_iterations=20))
    trainer.fit(model)


You can also use the block context manager within the forward function, or any of the step functions.

.. code-block:: python

    import lightning.pytorch as pl
    import poptorch


    class MyLightningModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(5, 10)
            self.layer2 = torch.nn.Linear(10, 5)
            self.layer3 = torch.nn.Linear(5, 5)
            self.layer4 = torch.nn.Linear(5, 5)

            self.act = torch.nn.ReLU()
            self.softmax = torch.nn.Softmax(dim=1)

        def forward(self, x):
            with poptorch.Block(ipu_id=0):
                x = self.act(self.layer1(x))

            with poptorch.Block(ipu_id=1):
                x = self.act(self.layer2(x))

            with poptorch.Block(ipu_id=2):
                x = self.act(self.layer3(x))
                x = self.act(self.layer4(x))

            with poptorch.Block(ipu_id=3):
                x = self.softmax(x)
            return x

        ...


    model = MyLightningModule()
    trainer = pl.Trainer(accelerator="ipu", devices=8, strategy=IPUStrategy(device_iterations=20))
    trainer.fit(model)
