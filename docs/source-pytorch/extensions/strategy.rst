###################
What is a Strategy?
###################

Strategy controls the model distribution across training, evaluation, and prediction to be used by the :doc:`Trainer <../common/trainer>`. It can be controlled by passing different
strategy with aliases (``"ddp"``, ``"ddp_spawn"``, ``"deepspeed"`` and so on) as well as a custom strategy to the ``strategy`` parameter for Trainer.

The Strategy in PyTorch Lightning handles the following responsibilities:

* Launch and teardown of training processes (if applicable).
* Setup communication between processes (NCCL, GLOO, MPI, and so on).
* Provide a unified communication interface for reduction, broadcast, and so on.
* Owns the :class:`~lightning.pytorch.core.LightningModule`
* Handles/owns optimizers and schedulers.


Strategy is a composition of one :doc:`Accelerator <../extensions/accelerator>`, one :ref:`Precision Plugin <extensions/plugins:Precision Plugins>`, a :ref:`CheckpointIO <extensions/plugins:CheckpointIO Plugins>`
plugin and other optional plugins such as the :ref:`ClusterEnvironment <extensions/plugins:Cluster Environments>`.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/strategies/overview.jpeg
    :alt: Illustration of the Strategy as a composition of the Accelerator and several plugins

We expose Strategies mainly for expert users that want to extend Lightning for new hardware support or new distributed backends (e.g. a backend not yet supported by `PyTorch <https://pytorch.org/docs/stable/distributed.html#backends>`_ itself).


----

*****************************
Selecting a Built-in Strategy
*****************************

Built-in strategies can be selected in two ways.

1. Pass the shorthand name to the ``strategy`` Trainer argument
2. Import a Strategy from :mod:`lightning.pytorch.strategies`, instantiate it and pass it to the ``strategy`` Trainer argument

The latter allows you to configure further options on the specific strategy.
Here are some examples:

.. code-block:: python

    # Training with the DistributedDataParallel strategy on 4 GPUs
    trainer = Trainer(strategy="ddp", accelerator="gpu", devices=4)

    # Training with the DistributedDataParallel strategy on 4 GPUs, with options configured
    trainer = Trainer(strategy=DDPStrategy(static_graph=True), accelerator="gpu", devices=4)

    # Training with the DDP Spawn strategy using auto accelerator selection
    trainer = Trainer(strategy="ddp_spawn", accelerator="auto", devices=4)

    # Training with the DeepSpeed strategy on available GPUs
    trainer = Trainer(strategy="deepspeed", accelerator="gpu", devices="auto")

    # Training with the DDP strategy using 3 CPU processes
    trainer = Trainer(strategy="ddp", accelerator="cpu", devices=3)

    # Training with the DDP Spawn strategy on 8 TPU cores
    trainer = Trainer(strategy="ddp_spawn", accelerator="tpu", devices=8)

The below table lists all relevant strategies available in Lightning with their corresponding short-hand name:

.. list-table:: Strategy Classes and Nicknames
   :widths: 20 20 20
   :header-rows: 1

   * - Name
     - Class
     - Description
   * - fsdp
     - :class:`~lightning.pytorch.strategies.FSDPStrategy`
     - Strategy for Fully Sharded Data Parallel training. :doc:`Learn more. <../advanced/model_parallel/fsdp>`
   * - ddp
     - :class:`~lightning.pytorch.strategies.DDPStrategy`
     - Strategy for multi-process single-device training on one or multiple nodes. :ref:`Learn more. <accelerators/gpu_intermediate:Distributed Data Parallel>`
   * - ddp_spawn
     - :class:`~lightning.pytorch.strategies.DDPStrategy`
     - Same as "ddp" but launches processes using ``torch.multiprocessing.spawn`` method and joins processes after training finishes. :ref:`Learn more. <accelerators/gpu_intermediate:Distributed Data Parallel Spawn>`
   * - deepspeed
     - :class:`~lightning.pytorch.strategies.DeepSpeedStrategy`
     - Provides capabilities to run training using the DeepSpeed library, with training optimizations for large billion parameter models. :doc:`Learn more. <../advanced/model_parallel/deepspeed>`
   * - hpu_parallel
     - ``HPUParallelStrategy``
     - Strategy for distributed training on multiple HPU devices. :doc:`Learn more. <../integrations/hpu/index>`
   * - hpu_single
     - ``SingleHPUStrategy``
     - Strategy for training on a single HPU device. :doc:`Learn more. <../integrations/hpu/index>`
   * - xla
     - :class:`~lightning.pytorch.strategies.XLAStrategy`
     - Strategy for training on multiple TPU devices using the :func:`torch_xla.distributed.xla_multiprocessing.spawn` method. :doc:`Learn more. <../accelerators/tpu>`
   * - single_xla
     - :class:`~lightning.pytorch.strategies.SingleXLAStrategy`
     - Strategy for training on a single XLA device, like TPUs. :doc:`Learn more. <../accelerators/tpu>`

----


**********************
Third-party Strategies
**********************

There are powerful third-party strategies that integrate well with Lightning but aren't maintained as part of the ``lightning`` package.
Checkout the gallery over :doc:`here <../integrations/strategies/index>`.

----


************************
Create a Custom Strategy
************************

Every strategy in Lightning is a subclass of one of the main base classes: :class:`~lightning.pytorch.strategies.Strategy`, :class:`~lightning.pytorch.strategies.SingleDeviceStrategy` or :class:`~lightning.pytorch.strategies.ParallelStrategy`.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/strategies/hierarchy.jpeg
    :alt: Strategy base classes

As an expert user, you may choose to extend either an existing built-in Strategy or create a completely new one by
subclassing the base classes.

.. code-block:: python

    from lightning.pytorch.strategies import DDPStrategy


    class CustomDDPStrategy(DDPStrategy):
        def configure_ddp(self):
            self.model = MyCustomDistributedDataParallel(
                self.model,
                device_ids=...,
            )

        def setup(self, trainer):
            # you can access the accelerator and plugins directly
            self.accelerator.setup()
            self.precision_plugin.connect(...)


The custom strategy can then be passed into the ``Trainer`` directly via the ``strategy`` parameter.

.. code-block:: python

    # custom strategy
    trainer = Trainer(strategy=CustomDDPStrategy())


Since the strategy also hosts the Accelerator and various plugins, you can customize all of them to work together as you like:

.. code-block:: python

    # custom strategy, with new accelerator and plugins
    accelerator = MyAccelerator()
    precision_plugin = MyPrecisionPlugin()
    strategy = CustomDDPStrategy(accelerator=accelerator, precision_plugin=precision_plugin)
    trainer = Trainer(strategy=strategy)
