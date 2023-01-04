#############
Fabric (Beta)
#############


:class:`~lightning_fabric.fabric.Fabric` library allows you to scale any PyTorch model with just a few lines of code!
With Fabric you can easily scale your model to run on distributed devices using the strategy of your choice, while keeping full control over the training loop and optimization logic.

With only a few changes to your code, Fabric allows you to:

- Automatic placement of models and data onto the device
- Automatic support for mixed precision (speedup and smaller memory footprint)
- Seamless switching between hardware (CPU, GPU, TPU)
- State-of-the-art distributed training strategies (DDP, FSDP, DeepSpeed)
- Easy-to-use launch command for spawning processes (DDP, torchelastic, etc)
- Multi-node support (TorchElastic, SLURM, and more)
- You keep full control of your training loop


.. code-block:: diff

      import torch
      import torch.nn as nn
      from torch.utils.data import DataLoader, Dataset

    + from lightning.fabric import Fabric

      class MyModel(nn.Module):
          ...

      class MyDataset(Dataset):
          ...

    + fabric = Fabric(accelerator="cuda", devices=8, strategy="ddp")
    + fabric.launch()

    - device = "cuda" if torch.cuda.is_available() else "cpu
      model = MyModel(...)
      optimizer = torch.optim.SGD(model.parameters())
    + model, optimizer = fabric.setup(model, optimizer)
      dataloader = DataLoader(MyDataset(...), ...)
    + dataloader = fabric.setup_dataloaders(dataloader)
      model.train()

      for epoch in range(num_epochs):
          for batch in dataloader:
    -         batch.to(device)
              optimizer.zero_grad()
              loss = model(batch)
    -         loss.backward()
    +         fabric.backward(loss)
              optimizer.step()


.. note:: :class:`~lightning_fabric.fabric.Fabric` is currently a beta feature. Its API is subject to change based on feedback.


----------

*****************
Convert to Fabric
*****************

Here are five easy steps to let :class:`~lightning_fabric.fabric.Fabric` scale your PyTorch models.

**Step 1:** Create the :class:`~lightning_fabric.fabric.Fabric` object at the beginning of your training code.

.. code-block:: python

    from lightning.fabric import Fabric

    fabric = Fabric()

**Step 2:** Call :meth:`~lightning_fabric.fabric.Fabric.setup` on each model and optimizer pair and :meth:`~lightning_fabric.fabric.Fabric.setup_dataloaders` on all your dataloaders.

.. code-block:: python

    model, optimizer = fabric.setup(model, optimizer)
    dataloader = fabric.setup_dataloaders(dataloader)

**Step 3:** Remove all ``.to`` and ``.cuda`` calls since :class:`~lightning_fabric.fabric.Fabric` will take care of it.

.. code-block:: diff

  - model.to(device)
  - batch.to(device)

**Step 4:** Replace ``loss.backward()`` by ``fabric.backward(loss)``.

.. code-block:: diff

  - loss.backward()
  + fabric.backward(loss)

**Step 5:** Run the script from the terminal with

.. code-block:: bash

    lightning run model path/to/train.py``

or use the :meth:`~lightning_fabric.fabric.Fabric.launch` method in a notebook.

|

That's it! You can now train on any device at any scale with a switch of a flag.
Check out our examples that use Fabric:

- `Image Classification <https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/image_classifier/README.md>`_
- `Generative Adversarial Network (GAN) <https://github.com/Lightning-AI/lightning/blob/master/examples/fabric/dcgan/README.md>`_


Here is how you run DDP with 8 GPUs and `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ precision:

.. code-block:: bash

    lightning run model ./path/to/train.py --strategy=ddp --devices=8 --accelerator=cuda --precision="bf16"

Or `DeepSpeed Zero3 <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`_ with mixed precision:

.. code-block:: bash

     lightning run model ./path/to/train.py --strategy=deepspeed --devices=8 --accelerator=cuda --precision=16

:class:`~lightning_fabric.fabric.Fabric` can also figure it out automatically for you!

.. code-block:: bash

    lightning run model ./path/to/train.py --devices=auto --accelerator=auto --precision=16


You can also easily use distributed collectives if required.

.. code-block:: python

    fabric = Fabric()

    # Transfer and concatenate tensors across processes
    fabric.all_gather(...)

    # Transfer an object from one process to all the others
    fabric.broadcast(..., src=...)

    # The total number of processes running across all devices and nodes.
    fabric.world_size

    # The global index of the current process across all devices and nodes.
    fabric.global_rank

    # The index of the current process among the processes running on the local node.
    fabric.local_rank

    # The index of the current node.
    fabric.node_rank

    # Whether this global rank is rank zero.
    if fabric.is_global_zero:
        # do something on rank 0
        ...

    # Wait for all processes to enter this call.
    fabric.barrier()


The code stays agnostic, whether you are running on CPU, on two GPUS or on multiple machines with many GPUs.

If you require custom data or model device placement, you can deactivate :class:`~lightning_fabric.fabric.Fabric`'s automatic placement by doing ``fabric.setup_dataloaders(..., move_to_device=False)`` for the data and ``fabric.setup(..., move_to_device=False)`` for the model.
Furthermore, you can access the current device from ``fabric.device`` or rely on :meth:`~lightning_fabric.fabric.Fabric.to_device` utility to move an object to the current device.


----------


Distributed Training Pitfalls
=============================

The :class:`~lightning_fabric.fabric.Fabric` provides you with the tools to scale your training, but there are several major challenges ahead of you now:


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


If you are facing one of those challenges, then you are already meeting the limit of :class:`~lightning_fabric.fabric.Fabric`.
We recommend you to convert to :doc:`Lightning <../starter/introduction>`, so you never have to worry about those.


----------

************
Fabric Flags
************

Fabric is specialized in accelerated distributed training and inference. It offers you convenient ways to configure
your device and communication strategy and to switch seamlessly from one to the other. The terminology and usage are
identical to Lightning, which means minimum effort for you to convert when you decide to do so.


accelerator
===========

Choose one of ``"cpu"``, ``"gpu"``, ``"tpu"``, ``"auto"`` (IPU support is coming soon).

.. code-block:: python

    # CPU accelerator
    fabric = Fabric(accelerator="cpu")

    # Running with GPU Accelerator using 2 GPUs
    fabric = Fabric(devices=2, accelerator="gpu")

    # Running with TPU Accelerator using 8 tpu cores
    fabric = Fabric(devices=8, accelerator="tpu")

    # Running with GPU Accelerator using the DistributedDataParallel strategy
    fabric = Fabric(devices=4, accelerator="gpu", strategy="ddp")

The ``"auto"`` option recognizes the machine you are on and selects the available accelerator.

.. code-block:: python

    # If your machine has GPUs, it will use the GPU Accelerator
    fabric = Fabric(devices=2, accelerator="auto")


strategy
========

Choose a training strategy: ``"dp"``, ``"ddp"``, ``"ddp_spawn"``, ``"tpu_spawn"``, ``"deepspeed"``, ``"ddp_sharded"``, or ``"ddp_sharded_spawn"``.

.. code-block:: python

    # Running with the DistributedDataParallel strategy on 4 GPUs
    fabric = Fabric(strategy="ddp", accelerator="gpu", devices=4)

    # Running with the DDP Spawn strategy using 4 cpu processes
    fabric = Fabric(strategy="ddp_spawn", accelerator="cpu", devices=4)


Additionally, you can pass in your custom strategy by configuring additional parameters.

.. code-block:: python

    from lightning.fabric.strategies import DeepSpeedStrategy

    fabric = Fabric(strategy=DeepSpeedStrategy(stage=2), accelerator="gpu", devices=2)


Support for Fully Sharded training strategies are coming soon.


devices
=======

Configure the devices to run on. Can be of type:

- int: the number of devices (e.g., GPUs) to train on
- list of int: which device index (e.g., GPU ID) to train on (0-indexed)
- str: a string representation of one of the above

.. code-block:: python

    # default used by Fabric, i.e., use the CPU
    fabric = Fabric(devices=None)

    # equivalent
    fabric = Fabric(devices=0)

    # int: run on two GPUs
    fabric = Fabric(devices=2, accelerator="gpu")

    # list: run on GPUs 1, 4 (by bus ordering)
    fabric = Fabric(devices=[1, 4], accelerator="gpu")
    fabric = Fabric(devices="1, 4", accelerator="gpu")  # equivalent

    # -1: run on all GPUs
    fabric = Fabric(devices=-1, accelerator="gpu")
    fabric = Fabric(devices="-1", accelerator="gpu")  # equivalent



gpus
====

.. warning:: ``gpus=x`` has been deprecated in v1.7 and will be removed in v2.0.
    Please use ``accelerator='gpu'`` and ``devices=x`` instead.

Shorthand for setting ``devices=X`` and ``accelerator="gpu"``.

.. code-block:: python

    # Run on two GPUs
    fabric = Fabric(accelerator="gpu", devices=2)

    # Equivalent
    fabric = Fabric(devices=2, accelerator="gpu")


tpu_cores
=========

.. warning:: ``tpu_cores=x`` has been deprecated in v1.7 and will be removed in v2.0.
    Please use ``accelerator='tpu'`` and ``devices=x`` instead.

Shorthand for ``devices=X`` and ``accelerator="tpu"``.

.. code-block:: python

    # Run on eight TPUs
    fabric = Fabric(accelerator="tpu", devices=8)

    # Equivalent
    fabric = Fabric(devices=8, accelerator="tpu")


num_nodes
=========


Number of cluster nodes for distributed operation.

.. code-block:: python

    # Default used by Fabric
    fabric = Fabric(num_nodes=1)

    # Run on 8 nodes
    fabric = Fabric(num_nodes=8)


Learn more about distributed multi-node training on clusters :doc:`here <../clouds/cluster>`.


precision
=========

Fabric supports double precision (64), full precision (32), or half precision (16) operation (including `bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_).
Half precision, or mixed precision, is the combined use of 32 and 16-bit floating points to reduce the memory footprint during model training.
This can result in improved performance, achieving significant speedups on modern GPUs.

.. code-block:: python

    # Default used by the Fabric
    fabric = Fabric(precision=32, devices=1)

    # 16-bit (mixed) precision
    fabric = Fabric(precision=16, devices=1)

    # 16-bit bfloat precision
    fabric = Fabric(precision="bf16", devices=1)

    # 64-bit (double) precision
    fabric = Fabric(precision=64, devices=1)


plugins
=======

:ref:`Plugins` allow you to connect arbitrary backends, precision libraries, clusters etc. For example:
To define your own behavior, subclass the relevant class and pass it in. Here's an example linking up your own
:class:`~lightning.fabric.plugins.environments.ClusterEnvironment`.

.. code-block:: python

    from lightning.fabric.plugins.environments import ClusterEnvironment


    class MyCluster(ClusterEnvironment):
        @property
        def main_address(self):
            return your_main_address

        @property
        def main_port(self):
            return your_main_port

        def world_size(self):
            return the_world_size


    fabric = Fabric(plugins=[MyCluster()], ...)


----------


**************
Fabric Methods
**************


setup
=====

Set up a model and corresponding optimizer(s). If you need to set up multiple models, call ``setup()`` on each of them.
Moves the model and optimizer to the correct device automatically.

.. code-block:: python

    model = nn.Linear(32, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Set up model and optimizer for accelerated training
    model, optimizer = fabric.setup(model, optimizer)

    # If you don't want Fabric to set the device
    model, optimizer = fabric.setup(model, optimizer, move_to_device=False)


The setup method also prepares the model for the selected precision choice so that operations during ``forward()`` get
cast automatically.

setup_dataloaders
=================

Set up one or multiple dataloaders for accelerated operation. If you are running a distributed strategy (e.g., DDP), Fabric
replaces the sampler automatically for you. In addition, the dataloader will be configured to move the returned
data tensors to the correct device automatically.

.. code-block:: python

    train_data = torch.utils.DataLoader(train_dataset, ...)
    test_data = torch.utils.DataLoader(test_dataset, ...)

    train_data, test_data = fabric.setup_dataloaders(train_data, test_data)

    # If you don't want Fabric to move the data to the device
    train_data, test_data = fabric.setup_dataloaders(train_data, test_data, move_to_device=False)

    # If you don't want Fabric to replace the sampler in the context of distributed training
    train_data, test_data = fabric.setup_dataloaders(train_data, test_data, replace_sampler=False)


backward
========

This replaces any occurrences of ``loss.backward()`` and makes your code accelerator and precision agnostic.

.. code-block:: python

    output = model(input)
    loss = loss_fn(output, target)

    # loss.backward()
    fabric.backward(loss)


to_device
=========

Use :meth:`~lightning_fabric.fabric.Fabric.to_device` to move models, tensors or collections of tensors to
the current device. By default :meth:`~lightning_fabric.fabric.Fabric.setup` and
:meth:`~lightning_fabric.fabric.Fabric.setup_dataloaders` already move the model and data to the correct
device, so calling this method is only necessary for manual operation when needed.

.. code-block:: python

    data = torch.load("dataset.pt")
    data = fabric.to_device(data)


seed_everything
===============

Make your code reproducible by calling this method at the beginning of your run.

.. code-block:: python

    # Instead of `torch.manual_seed(...)`, call:
    fabric.seed_everything(1234)


This covers PyTorch, NumPy and Python random number generators. In addition, Fabric takes care of properly initializing
the seed of dataloader worker processes (can be turned off by passing ``workers=False``).


autocast
========

Let the precision backend autocast the block of code under this context manager. This is optional and already done by
Fabric for the model's forward method (once the model was :meth:`~lightning_fabric.fabric.Fabric.setup`).
You need this only if you wish to autocast more operations outside the ones in model forward:

.. code-block:: python

    model, optimizer = fabric.setup(model, optimizer)

    # Fabric handles precision automatically for the model
    output = model(inputs)

    with fabric.autocast():  # optional
        loss = loss_function(output, target)

    fabric.backward(loss)
    ...


print
=====

Print to the console via the built-in print function, but only on the main process.
This avoids excessive printing and logs when running on multiple devices/nodes.


.. code-block:: python

    # Print only on the main process
    fabric.print(f"{epoch}/{num_epochs}| Train Epoch Loss: {loss}")


save
====

Save contents to a checkpoint. Replaces all occurrences of ``torch.save(...)`` in your code. Fabric will take care of
handling the saving part correctly, no matter if you are running a single device, multi-devices or multi-nodes.

.. code-block:: python

    # Instead of `torch.save(...)`, call:
    fabric.save(model.state_dict(), "path/to/checkpoint.ckpt")


load
====

Load checkpoint contents from a file. Replaces all occurrences of ``torch.load(...)`` in your code. Fabric will take care of
handling the loading part correctly, no matter if you are running a single device, multi-device, or multi-node.

.. code-block:: python

    # Instead of `torch.load(...)`, call:
    fabric.load("path/to/checkpoint.ckpt")


barrier
=======

Call this if you want all processes to wait and synchronize. Once all processes have entered this call,
execution continues. Useful for example when you want to download data on one process and make all others wait until
the data is written to disk.

.. code-block:: python

    # Download data only on one process
    if fabric.global_rank == 0:
        download_data("http://...")

    # Wait until all processes meet up here
    fabric.barrier()

    # All processes are allowed to read the data now


no_backward_sync
================

Use this context manager when performing gradient accumulation and using a distributed strategy (e.g., DDP).
It will speed up your training loop by cutting redundant communication between processes during the accumulation phase.

.. code-block:: python

    # Accumulate gradient 8 batches at a time
    is_accumulating = batch_idx % 8 != 0

    with fabric.no_backward_sync(model, enabled=is_accumulating):
        output = model(input)
        loss = ...
        fabric.backward(loss)
        ...

    # Step the optimizer every 8 batches
    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()

Both the model's `.forward()` and the `fabric.backward()` call need to run under this context as shown in the example above.
For single-device strategies, it is a no-op. There are strategies that don't support this:

- deepspeed
- dp
- xla

For these, the context manager falls back to a no-op and emits a warning.
