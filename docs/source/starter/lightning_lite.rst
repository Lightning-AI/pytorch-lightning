###########################################
LightningLite - Stepping Stone to Lightning
###########################################


:class:`~pytorch_lightning.lite.LightningLite` enables pure PyTorch users to scale their existing code
on any kind of device while retaining full control over their own loops and optimization logic.

.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/lite/lightning_lite.gif
    :alt: Animation showing how to convert your PyTorch code to LightningLite.
    :width: 500
    :align: center

|

:class:`~pytorch_lightning.lite.LightningLite` is the right tool for you if you match one of the two following descriptions:

- I want to quickly scale my existing code to multiple devices with minimal code changes.
- I would like to convert my existing code to the Lightning API, but a full path to Lightning transition might be too complex. I am looking for a stepping stone to ensure reproducibility during the transition.


.. warning:: :class:`~pytorch_lightning.lite.LightningLite` is currently a beta feature. Its API is subject to change based on your feedback.


----------

****************
Learn by example
****************


My Existing PyTorch Code
========================

The ``run`` function contains custom training loop used to train ``MyModel`` on ``MyDataset`` for ``num_epochs`` epochs.

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset


    class MyModel(nn.Module):
        ...


    class MyDataset(Dataset):
        ...


    def run(args):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MyModel(...).to(device)
        optimizer = torch.optim.SGD(model.parameters(), ...)

        dataloader = DataLoader(MyDataset(...), ...)

        model.train()
        for epoch in range(args.num_epochs):
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()


    run(args)


----------


Convert to LightningLite
========================

Here are five required steps to convert to :class:`~pytorch_lightning.lite.LightningLite`.

1. Subclass :class:`~pytorch_lightning.lite.LightningLite` and override its :meth:`~pytorch_lightning.lite.LightningLite.run` method.
2. Move the body of your existing ``run`` function into :class:`~pytorch_lightning.lite.LightningLite` ``run`` method.
3. Remove all ``.to(...)``, ``.cuda()`` etc calls since :class:`~pytorch_lightning.lite.LightningLite` will take care of it.
4. Apply :meth:`~pytorch_lightning.lite.LightningLite.setup` over each model and optimizers pair and :meth:`~pytorch_lightning.lite.LightningLite.setup_dataloaders` on all your dataloaders and replace ``loss.backward()`` by ``self.backward(loss)``.
5. Instantiate your :class:`~pytorch_lightning.lite.LightningLite` subclass and call its :meth:`~pytorch_lightning.lite.LightningLite.run` method.

|

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from pytorch_lightning.lite import LightningLite


    class MyModel(nn.Module):
        ...


    class MyDataset(Dataset):
        ...


    class Lite(LightningLite):
        def run(self, args):

            model = MyModel(...)
            optimizer = torch.optim.SGD(model.parameters(), ...)
            model, optimizer = self.setup(model, optimizer)  # Scale your model / optimizers

            dataloader = DataLoader(MyDataset(...), ...)
            dataloader = self.setup_dataloaders(dataloader)  # Scale your dataloaders

            model.train()
            for epoch in range(args.num_epochs):
                for batch in dataloader:
                    optimizer.zero_grad()
                    loss = model(batch)
                    self.backward(loss)  # instead of loss.backward()
                    optimizer.step()


    Lite(...).run(args)


That's all. You can now train on any kind of device and scale your training.

:class:`~pytorch_lightning.lite.LightningLite` takes care of device management, so you don't have to.
You should remove any device-specific logic within your code.

Here is how to train on eight GPUs with `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ precision:

.. code-block:: python

    Lite(strategy="ddp", devices=8, accelerator="gpu", precision="bf16").run(10)

Here is how to use `DeepSpeed Zero3 <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`_ with eight GPUs and precision 16:

.. code-block:: python

    Lite(strategy="deepspeed", devices=8, accelerator="gpu", precision=16).run(10)

:class:`~pytorch_lightning.lite.LightningLite` can also figure it out automatically for you!

.. code-block:: python

    Lite(devices="auto", accelerator="auto", precision=16).run(10)

You can also easily use distributed collectives if required.
Here is an example while running on 256 GPUs (eight GPUs times 32 nodes).

.. code-block:: python

    class Lite(LightningLite):
        def run(self):

            # Transfer and concatenate tensors across processes
            self.all_gather(...)

            # Transfer an object from one process to all the others
            self.broadcast(..., src=...)

            # The total number of processes running across all devices and nodes.
            self.world_size

            # The global index of the current process across all devices and nodes.
            self.global_rank

            # The index of the current process among the processes running on the local node.
            self.local_rank

            # The index of the current node.
            self.node_rank

            # Wether this global rank is rank zero.
            if self.is_global_zero:
                # do something on rank 0
                ...

            # Wait for all processes to enter this call.
            self.barrier()


    Lite(strategy="ddp", gpus=8, num_nodes=32, accelerator="gpu").run()


If you require custom data or model device placement, you can deactivate
:class:`~pytorch_lightning.lite.LightningLite` automatic placement by doing
``self.setup_dataloaders(..., move_to_device=False)`` for the data and
``self.setup(..., move_to_device=False)`` for the model.
Furthermore, you can access the current device from ``self.device`` or
rely on :meth:`~pytorch_lightning.lite.LightningLite.to_device`
utility to move an object to the current device.


.. note:: We recommend instantiating the models within the :meth:`~pytorch_lightning.lite.LightningLite.run` method as large models would cause an out-of-memory error otherwise.

.. tip::

    If you have hundreds or thousands of lines within your :meth:`~pytorch_lightning.lite.LightningLite.run` function
    and you are feeling unsure about them, then that is the correct feeling.
    In 2019, our :class:`~pytorch_lightning.core.lightning.LightningModule` was getting larger
    and we got the same feeling, so we started to organize our code for simplicity, interoperability and standardization.
    This is definitely a good sign that you should consider refactoring your code and / or switching to
    :class:`~pytorch_lightning.core.lightning.LightningModule` ultimately.


----------


Distributed Training Pitfalls
=============================

The :class:`~pytorch_lightning.lite.LightningLite` provides you with the tools to scale your training,
but there are several major challenges ahead of you now:


.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - Processes divergence
     - This happens when processes execute a different section of the code due to different if/else conditions, race conditions on existing files and so on, resulting in hanging.
   * - Cross processes reduction
     - Miscalculated metrics or gradients due to errors in their reduction.
   * - Large sharded models
     - Instantiation, materialization and state management of large models.
   * - Rank 0 only actions
     - Logging, profiling, and so on.
   * - Checkpointing / Early stopping / Callbacks / Logging
     - Ability to customize your training behavior easily and make it stateful.
   * - Fault-tolerant training
     - Ability to resume from a failure as if it never happened.


If you are facing one of those challenges, then you are already meeting the limit of :class:`~pytorch_lightning.lite.LightningLite`.
We recommend you to convert to :doc:`Lightning <../starter/introduction>`, so you never have to worry about those.

----------

Convert to Lightning
====================

:class:`~pytorch_lightning.lite.LightningLite` is a stepping stone to transition fully to the Lightning API and benefit
from its hundreds of features.

You can see our :class:`~pytorch_lightning.lite.LightningLite` class as a
future :class:`~pytorch_lightning.core.lightning.LightningModule`, and slowly refactor your code into its API.
Below, the :meth:`~pytorch_lightning.core.lightning.LightningModule.training_step`, :meth:`~pytorch_lightning.core.lightning.LightningModule.forward`,
:meth:`~pytorch_lightning.core.lightning.LightningModule.configure_optimizers`, :meth:`~pytorch_lightning.core.lightning.LightningModule.train_dataloader` methods
are implemented.


.. code-block:: python

    class Lite(LightningLite):

        # 1. This would become the LightningModule `__init__` function.
        def run(self, args):
            self.args = args

            self.model = MyModel(...)

            self.fit()  # This would be automated by the Lightning Trainer.

        # 2. This can be fully removed as Lightning creates its own fitting loop,
        # and sets up the model, optimizer, dataloader, etc for you.
        def fit(self):
            # setup everything
            optimizer = self.configure_optimizers()
            self.model, optimizer = self.setup(self.model, optimizer)
            dataloader = self.setup_dataloaders(self.train_dataloader())

            # start fitting
            self.model.train()
            for epoch in range(num_epochs):
                for batch in enumerate(dataloader):
                    optimizer.zero_grad()
                    loss = self.training_step(batch, batch_idx)
                    self.backward(loss)
                    optimizer.step()

        # 3. This stays here as it belongs to the LightningModule.
        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            return self.forward(batch)

        def configure_optimizers(self):
            return torch.optim.SGD(self.model.parameters(), ...)

        # 4. [Optionally] This can stay here or be extracted to the LightningDataModule to enable higher composability.
        def train_dataloader(self):
            return DataLoader(MyDataset(...), ...)


    Lite(...).run(args)


Finally, change the :meth:`~pytorch_lightning.lite.LightningLite.run` into a
:meth:`~pytorch_lightning.core.lightning.LightningModule.__init__` and drop the ``fit`` call from inside.

.. code-block:: python

    from pytorch_lightning import LightningDataModule, LightningModule, Trainer


    class LightningModel(LightningModule):
        def __init__(self, args):
            super().__init__()
            self.model = MyModel(...)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            loss = self(batch)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.model.parameters(), lr=0.001)


    class BoringDataModule(LightningDataModule):
        def train_dataloader(self):
            return DataLoader(MyDataset(...), ...)


    trainer = Trainer(max_epochs=10)
    trainer.fit(LightningModel(), datamodule=BoringDataModule())


You have successfully converted to PyTorch Lightning, and can now benefit from its hundred of features!

----------

********************
Lightning Lite Flags
********************

Lite is specialized in accelerated distributed training and inference. It offers you convenient ways to configure
your device and communication strategy and to switch seamlessly from one to the other. The terminology and usage are
identical to Lightning, which means minimum effort for you to convert when you decide to do so.


accelerator
===========

Choose one of ``"cpu"``, ``"gpu"``, ``"tpu"``, ``"auto"`` (IPU support is coming soon).

.. code-block:: python

    # CPU accelerator
    lite = Lite(accelerator="cpu")

    # Running with GPU Accelerator using 2 GPUs
    lite = Lite(devices=2, accelerator="gpu")

    # Running with TPU Accelerator using 8 tpu cores
    lite = Lite(devices=8, accelerator="tpu")

    # Running with GPU Accelerator using the DistributedDataParallel strategy
    lite = Lite(devices=4, accelerator="gpu", strategy="ddp")

The ``"auto"`` option recognizes the machine you are on and selects the available accelerator.

.. code-block:: python

    # If your machine has GPUs, it will use the GPU Accelerator
    lite = Lite(devices=2, accelerator="auto")


strategy
========

Choose a training strategy: ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"tpu_spawn"``, ``"deepspeed"``, ``"ddp_sharded"``, or ``"ddp_sharded_spawn"``.

.. code-block:: python

    # Running with the DistributedDataParallel strategy on 4 GPUs
    lite = Lite(strategy="ddp", accelerator="gpu", devices=4)

    # Running with the DDP Spawn strategy using 4 cpu processes
    lite = Lite(strategy="ddp_spawn", accelerator="cpu", devices=4)


Additionally, you can pass in your custom training type strategy by configuring additional parameters.

.. code-block:: python

    from pytorch_lightning.strategies import DeepSpeedStrategy

    lite = Lite(strategy=DeepSpeedStrategy(stage=2), accelerator="gpu", devices=2)


Support for Horovod and Fully Sharded training strategies are coming soon.


devices
=======

Configure the devices to run on. Can be of type:

- int: the number of devices (e.g., GPUs) to train on
- list of int: which device index (e.g., GPU ID) to train on (0-indexed)
- str: a string representation of one of the above

.. code-block:: python

    # default used by Lite, i.e., use the CPU
    lite = Lite(devices=None)

    # equivalent
    lite = Lite(devices=0)

    # int: run on two GPUs
    lite = Lite(devices=2, accelerator="gpu")

    # list: run on GPUs 1, 4 (by bus ordering)
    lite = Lite(devices=[1, 4], accelerator="gpu")
    lite = Lite(devices="1, 4", accelerator="gpu")  # equivalent

    # -1: run on all GPUs
    lite = Lite(devices=-1, accelerator="gpu")
    lite = Lite(devices="-1", accelerator="gpu")  # equivalent



gpus
====

Shorthand for setting ``devices=X`` and ``accelerator="gpu"``.

.. code-block:: python

    # Run on two GPUs
    lite = Lite(accelerator="gpu", devices=2)

    # Equivalent
    lite = Lite(devices=2, accelerator="gpu")


tpu_cores
=========

Shorthand for ``devices=X`` and ``accelerator="tpu"``.

.. code-block:: python

    # Run on eight TPUs
    lite = Lite(accelerator="tpu", devices=8)

    # Equivalent
    lite = Lite(devices=8, accelerator="tpu")


num_nodes
=========


Number of cluster nodes for distributed operation.

.. code-block:: python

    # Default used by Lite
    lite = Lite(num_nodes=1)

    # Run on 8 nodes
    lite = Lite(num_nodes=8)


Learn more about distributed multi-node training on clusters :doc:`here <../clouds/cluster>`.


precision
=========

Lightning Lite supports double precision (64), full precision (32), or half precision (16) operation (including `bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_).
Half precision, or mixed precision, is the combined use of 32 and 16-bit floating points to reduce the memory footprint during model training.
This can result in improved performance, achieving significant speedups on modern GPUs.

.. code-block:: python

    # Default used by the Lite
    lite = Lite(precision=32, devices=1)

    # 16-bit (mixed) precision
    lite = Lite(precision=16, devices=1)

    # 16-bit bfloat precision
    lite = Lite(precision="bf16", devices=1)

    # 64-bit (double) precision
    lite = Lite(precision=64, devices=1)


plugins
=======

:ref:`Plugins` allow you to connect arbitrary backends, precision libraries, clusters etc. For example:
To define your own behavior, subclass the relevant class and pass it in. Here's an example linking up your own
:class:`~pytorch_lightning.plugins.environments.ClusterEnvironment`.

.. code-block:: python

    from pytorch_lightning.plugins.environments import ClusterEnvironment


    class MyCluster(ClusterEnvironment):
        @property
        def main_address(self):
            return your_main_address

        @property
        def main_port(self):
            return your_main_port

        def world_size(self):
            return the_world_size


    lite = Lite(plugins=[MyCluster()], ...)


----------


**********************
Lightning Lite Methods
**********************


run
===

The run method serves two purposes:

1.  Override this method from the :class:`~pytorch_lightning.lite.lite.LightningLite` class and put your
    training (or inference) code inside.
2.  Launch the training procedure by calling the run method. Lite will take care of setting up the distributed backend.

You can optionally pass arguments to the run method. For example, the hyperparameters or a backbone for the model.

.. code-block:: python

    from pytorch_lightning.lite import LightningLite


    class Lite(LightningLite):

        # Input arguments are optional; put whatever you need
        def run(self, learning_rate, num_layers):
            """Here goes your training loop"""


    lite = Lite(accelerator="gpu", devices=2)
    lite.run(learning_rate=0.01, num_layers=12)


setup
=====

Set up a model and corresponding optimizer(s). If you need to set up multiple models, call ``setup()`` on each of them.
Moves the model and optimizer to the correct device automatically.

.. code-block:: python

    model = nn.Linear(32, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Set up model and optimizer for accelerated training
    model, optimizer = self.setup(model, optimizer)

    # If you don't want Lite to set the device
    model, optimizer = self.setup(model, optimizer, move_to_device=False)


The setup method also prepares the model for the selected precision choice so that operations during ``forward()`` get
cast automatically.

setup_dataloaders
=================

Set up one or multiple dataloaders for accelerated operation. If you are running a distributed strategy (e.g., DDP), Lite
replaces the sampler automatically for you. In addition, the dataloader will be configured to move the returned
data tensors to the correct device automatically.

.. code-block:: python

    train_data = torch.utils.DataLoader(train_dataset, ...)
    test_data = torch.utils.DataLoader(test_dataset, ...)

    train_data, test_data = self.setup_dataloaders(train_data, test_data)

    # If you don't want Lite to move the data to the device
    train_data, test_data = self.setup_dataloaders(train_data, test_data, move_to_device=False)

    # If you don't want Lite to replace the sampler in the context of distributed training
    train_data, test_data = self.setup_dataloaders(train_data, test_data, replace_sampler=False)


backward
========

This replaces any occurrences of ``loss.backward()`` and makes your code accelerator and precision agnostic.

.. code-block:: python

    output = model(input)
    loss = loss_fn(output, target)

    # loss.backward()
    self.backward(loss)


to_device
=========

Use :meth:`~pytorch_lightning.lite.lite.LightningLite.to_device` to move models, tensors or collections of tensors to
the current device. By default :meth:`~pytorch_lightning.lite.lite.LightningLite.setup` and
:meth:`~pytorch_lightning.lite.lite.LightningLite.setup_dataloaders` already move the model and data to the correct
device, so calling this method is only necessary for manual operation when needed.

.. code-block:: python

    data = torch.load("dataset.pt")
    data = self.to_device(data)


seed_everything
===============

Make your code reproducible by calling this method at the beginning of your run.

.. code-block:: python

    # Instead of `torch.manual_seed(...)`, call:
    self.seed_everything(1234)


This covers PyTorch, NumPy and Python random number generators. In addition, Lite takes care of properly initializing
the seed of dataloader worker processes (can be turned off by passing ``workers=False``).


autocast
========

Let the precision backend autocast the block of code under this context manager. This is optional and already done by
Lite for the model's forward method (once the model was :meth:`~pytorch_lightning.lite.lite.LightningLite.setup`).
You need this only if you wish to autocast more operations outside the ones in model forward:

.. code-block:: python

    model, optimizer = self.setup(model, optimizer)

    # Lite handles precision automatically for the model
    output = model(inputs)

    with self.autocast():  # optional
        loss = loss_function(output, target)

    self.backward(loss)
    ...


print
=====

Print to the console via the built-in print function, but only on the main process.
This avoids excessive printing and logs when running on multiple devices/nodes.


.. code-block:: python

    # Print only on the main process
    self.print(f"{epoch}/{num_epochs}| Train Epoch Loss: {loss}")


save
====

Save contents to a checkpoint. Replaces all occurrences of ``torch.save(...)`` in your code. Lite will take care of
handling the saving part correctly, no matter if you are running a single device, multi-devices or multi-nodes.

.. code-block:: python

    # Instead of `torch.save(...)`, call:
    self.save(model.state_dict(), "path/to/checkpoint.ckpt")


load
====

Load checkpoint contents from a file. Replaces all occurrences of ``torch.load(...)`` in your code. Lite will take care of
handling the loading part correctly, no matter if you are running a single device, multi-device, or multi-node.

.. code-block:: python

    # Instead of `torch.load(...)`, call:
    self.load("path/to/checkpoint.ckpt")


barrier
=======

Call this if you want all processes to wait and synchronize. Once all processes have entered this call,
execution continues. Useful for example when you want to download data on one process and make all others wait until
the data is written to disk.

.. code-block:: python

    # Download data only on one process
    if self.global_rank == 0:
        download_data("http://...")

    # Wait until all processes meet up here
    self.barrier()

    # All processes are allowed to read the data now
