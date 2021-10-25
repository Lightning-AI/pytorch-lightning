###########################################
LightningLite - Stepping Stone to Lightning
###########################################


.. image:: https://pl-public-data.s3.amazonaws.com/docs/static/images/lite/lightning_lite.gif
    :alt: Animation showing how to convert a standard training loop to a Lightning loop
    :width: 600px
    :align: center

|

:class:`~pytorch_lightning.lite.LightningLite` enables pure PyTorch users to scale their existing code
on any kind of device while retaining full control over their own loops and optimization logic.

:class:`~pytorch_lightning.lite.LightningLite` is the right tool for you if you match one of the two following descriptions:

- I want to quickly scale my existing code to multiple devices with minimal code changes.
- I would like to convert my existing code to the Lightning API, but a full path to Lightning transition might be too complex. I am looking for a stepping stone to ensure reproducibility during the transition.


----------


****************
Learn by example
****************


My existing PyTorch code
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


    def run(num_epochs: int):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = MyModel(...).to(device)
        optimizer = torch.optim.SGD(model.parameters(), ...)

        dataloader = DataLoader(MyDataset(...), ...)

        model.train()
        for epoch in range(num_epochs):
            for batch in dataloader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss = model(batch)
                loss.backward()
                optimizer.step()


    run(10)


----------


Convert to LightningLite
========================

Here are 4 required steps to convert to :class:`~pytorch_lightning.lite.LightningLite`.

1. Subclass :class:`~pytorch_lightning.lite.LightningLite` and override its :meth:`~pytorch_lightning.lite.LightningLite.run` method.
2. Move the body of your existing `run` function.
3. Apply :meth:`~pytorch_lightning.lite.LightningLite.setup` over each model and optimizers pair, :meth:`~pytorch_lightning.lite.LightningLite.setup_dataloaders` on all your dataloaders and replace ``loss.backward()`` by ``self.backward(loss)``
4. Instantiate your :class:`~pytorch_lightning.lite.LightningLite` and call its :meth:`~pytorch_lightning.lite.LightningLite.run` method.


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
        def run(self, num_epochs: int):

            model = MyModel(...)
            optimizer = torch.optim.SGD(model.parameters(), ...)

            model, optimizer = self.setup(model, optimizer)

            dataloader = DataLoader(MyDataset(...), ...)
            dataloader = self.setup_dataloaders(dataloader)

            model.train()
            for epoch in range(num_epochs):
                for batch in dataloader:
                    optimizer.zero_grad()
                    loss = model(batch)
                    self.backward(loss)
                    optimizer.step()


    Lite(...).run(10)


That's all. You can now train on any kind of device and scale your training.
The :class:`~pytorch_lightning.lite.LightningLite` takes care of device management, so you don't have to.
You should remove any device specific logic within your code.
Here is how to train on 8 GPUs with `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ precision:

.. code-block:: python

    Lite(strategy="ddp", devices=8, accelerator="gpu", precision="bf16").run(10)

Here is how to use `DeepSpeed Zero3 <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`_ with 8 GPUs and precision 16:

.. code-block:: python

    Lite(strategy="deepspeed", devices=8, accelerator="gpu", precision=16).run(10)

Lightning can also figure it automatically for you!

.. code-block:: python

    Lite(devices="auto", accelerator="auto", precision=16).run(10)


You can also easily use distributed collectives if required.
Here is an example while running on 256 GPUs.

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

            # Reduce a boolean decision across processes.
            self.reduce_decision(...)


    Lite(strategy="ddp", gpus=8, num_nodes=32, accelerator="gpu").run()


.. note:: We recommend instantiating the models within the :meth:`~pytorch_lightning.lite.LightningLite.run` method as large models would cause OOM Error otherwise.


----------


Distributed Training Pitfalls
=============================

The :class:`~pytorch_lightning.lite.LightningLite` provides you only with the tool to scale your training,
but there are several major challenges ahead of you now:


.. list-table::
   :widths: 50 50
   :header-rows: 0

   * - Processes divergence
     - This happens when processes execute different section of the code due to different if/else condition, race condition on existing files, etc., resulting in hanging.
   * - Cross processes reduction
     - Wrongly reported metrics or gradients due mis-reduction.
   * - Large sharded models
     - Instantiation, materialization and state management of large models.
   * - Rank 0 only actions
     - Logging, profiling, etc.
   * - Checkpointing / Early stopping / Callbacks
     - Ability to easily customize your training behaviour and make it stateful.
   * - Batch-level fault tolerance training
     - Ability to resume from a failure as if it never happened.


If you are facing one of those challenges then you are already meeting the limit of :class:`~pytorch_lightning.lite.LightningLite`.
We recommend you to convert to :doc:`Lightning <../starter/new-project>`, so you never have to worry about those.

----------

Convert to Lightning
====================

The :class:`~pytorch_lightning.lite.LightningLite` is a stepping stone to transition fully to the Lightning API and benefits
from its hundreds of features.

.. code-block:: python

    from pytorch_lightning import LightningDataModule, LightningModule, Trainer


    class LiftModel(LightningModule):
        def __init__(self, module: nn.Module):
            super().__init__()
            self.module = module

        def forward(self, x):
            return self.module(x)

        def training_step(self, batch, batch_idx):
            loss = self(batch)
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch, batch_idx):
            loss = self(batch)
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.001)


    class BoringDataModule(LightningDataModule):
        def __init__(self, dataset: Dataset):
            super().__init__()
            self.dataset = dataset

        def train_dataloader(self):
            return DataLoader(self.dataset)


    seed_everything(42)
    model = MyModel(...)
    lightning_module = LiftModel(model)
    dataset = MyDataset(...)
    datamodule = BoringDataModule(dataset)
    trainer = Trainer(max_epochs=10)
    trainer.fit(lightning_module, datamodule=datamodule)


----------


********************
Lightning Lite Flags
********************


Lite is a specialist for accelerated distributed training and inference. It offers you convenient ways to configure
your device and communication strategy and to seamlessly switch from one to the other. The terminology and usage is
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

The ``"auto"`` option recognizes the machine you are on, and selects the available accelerator.

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

    from pytorch_lightning.plugins import DeepSpeedPlugin

    lite = Lite(strategy=DeepSpeedPlugin(stage=2), accelerator="gpu", devices=2)


Support for Horovod and Fully Sharded training strategies are coming soon.


devices
=======

Configure the devices to run on. Can of type:

- int: the number of GPUs to train on
- list of int: which GPUs to train on (0-indexed)
- str: a string representation of one of the above

.. code-block:: python

    # default used by Lite, i.e., use the CPU
    lite = Lite(devices=None)

    # equivalent
    lite = Lite(devices=0)

    # int: run on 2 GPUs
    lite = Lite(devices=2, accelerator="gpu")

    # list: run on GPUs 1, 4 (by bus ordering)
    lite = Lite(devices=[1, 4], accelerator="gpu")
    lite = Lite(devices="1, 4", accelerator="gpu")  # equivalent

    # -1: run on all GPUs
    lite = Lite(devices=-1)
    lite = Lite(devices="-1")  # equivalent



gpus
====

Shorthand for setting ``devices=X`` and ``accelerator="gpu"``.

.. code-block:: python

    # Run on 2 GPUs
    lite = Lite(gpus=2)

    # Equivalent
    lite = Lite(devices=2, accelerator="gpu")


tpu_cores
=========

Shorthand for ``devices=X`` and ``accelerator="tpu"``.

.. code-block:: python

    # Run on 8 TPUs
    lite = Lite(gpus=8)

    # Equivalent
    lite = Lite(devices=8, accelerator="tpu")


num_nodes
=========


Number of cluster nodes for distributed operation.

.. testcode::

    # Default used by Lite
    lite = Lite(num_nodes=1)

    # Run on 8 nodes
    lite = Lite(num_nodes=8)


Learn more about distributed multi-node training on clusters :doc:`here <../clouds/cluster>`.


precision
=========

Lightning Lite supports double precision (64), full precision (32), or half precision (16) operation (including `bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_).
Half precision, or mixed precision, is the combined use of 32 and 16 bit floating points to reduce memory footprint during model training.
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

The run method servers two purposes:

1.  Override this method from the :class:`~pytorch_lightning.lite.lite.LightningLite` class and put your
    training (or inference) code inside.
2.  Launch the training by calling the run method. Lite will take care of setting up the distributed backend.

You can optionally pass arguments to the run method. For example, the hyperparameters or a backbone for the model.

.. code-block:: python

    from pytorch_lightning.lite import LightningLite

    class Lite(LightningLite):

        # Input arguments are optional, put whatever you need
        def run(self, learning_rate, num_layers):
            # Here goes your training loop

    lite = Lite(accelerator="gpu", devices=2)
    lite.run(learning_rate=0.01, num_layers=12)


setup
=====

Setup a model and corresponding optimizer(s). If you need to setup multiple models, call ``setup()`` on each of them.
Moves the model and optimizer to the correct device automatically.

.. code-block:: python

    model = nn.Linear(32, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Setup model and optimizer for accelerated training
    model, optimizer = self.setup(model, optimizer)

    # If you don't want Lite to set the device
    model, optimizer = self.setup(model, optimizer, move_to_device=False)


The setup method also prepares the model for the selected precision choice so that operations during ``forward()`` get
cast automatically.

setup_dataloaders
=================

Setup one or multiple dataloaders for accelerated operation. If you are running a distributed plugin (e.g., DDP), Lite
will replace the sampler automatically for you. In addition, the dataloader will be configured to move the returned
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

This replaces any occurences of ``loss.backward()`` and will make your code accelerator and precision agnostic.

.. code-block:: python

    output = model(input)
    loss = loss_fn(output, target)

    # loss.backward()
    self.backward(loss)


to_device
=========

Use :class:`~pytorch_lightning.lite.lite.LightningLite.to_device` to move models, tensors or collections of tensors to
the current device. By default :class:`~pytorch_lightning.lite.lite.LightningLite.setup` and
:class:`~pytorch_lightning.lite.lite.LightningLite.setup_dataloaders` already move the model and data to the correct
device, so calling this method is only necessary for manual operation when needed.

.. code-block:: python

    data = torch.load("dataset.pt")
    data = self.to_device(data)


print
=====

Print to the console via the built-in print function, but only on the main process.


.. code-block:: python

    # Print only on the main process
    self.print(f"{epoch}/{num_epochs}| Train Epoch Loss: {loss}")
