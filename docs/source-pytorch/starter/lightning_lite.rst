##############
Lightning Lite
##############


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

The ``train`` function contains a standard training loop used to train ``MyModel`` on ``MyDataset`` for ``num_epochs`` epochs.

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset


    class MyModel(nn.Module):
        ...


    class MyDataset(Dataset):
        ...


    def train(args):
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


    train(args)

----------


Convert to LightningLite
========================

Here are five easy steps to let :class:`~pytorch_lightning.lite.LightningLite` scale your PyTorch models.

1. Create the :class:`~pytorch_lightning.lite.LightningLite` object at the beginning of your training code.
2. Remove all ``.to`` and ``.cuda`` calls since :class:`~pytorch_lightning.lite.LightningLite` will take care of it.
3. Apply :meth:`~pytorch_lightning.lite.LightningLite.setup` over each model and optimizers pair and :meth:`~pytorch_lightning.lite.LightningLite.setup_dataloaders` on all your dataloaders and replace ``loss.backward()`` by ``lite.backward(loss)``.
4. Run the script from the terminal using ``lightning run model path/to/train.py`` or use the :meth:`~pytorch_lightning.lite.LightningLite.launch` method in a notebook.

|

.. code-block:: python

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
    from lightning.lite import LightningLite


    class MyModel(nn.Module):
        ...


    class MyDataset(Dataset):
        ...


    def train(args):

        lite = LightningLite()

        model = MyModel(...)
        optimizer = torch.optim.SGD(model.parameters(), ...)
        model, optimizer = lite.setup(model, optimizer)  # Scale your model / optimizers

        dataloader = DataLoader(MyDataset(...), ...)
        dataloader = lite.setup_dataloaders(dataloader)  # Scale your dataloaders

        model.train()
        for epoch in range(args.num_epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                loss = model(batch)
                lite.backward(loss)  # instead of loss.backward()
                optimizer.step()


    train(args)


That's all you need to do to your code. You can now train on any kind of device and scale your training.
Check out `this <https://github.com/Lightning-AI/lightning/blob/master/examples/lite/image_classifier_2_lite.py>`_ full MNIST training example with LightningLite.

Here is how to train on eight GPUs with `torch.bfloat16 <https://pytorch.org/docs/1.10.0/generated/torch.Tensor.bfloat16.html>`_ precision:

.. code-block:: bash

    lightning run model ./path/to/train.py --strategy=ddp --devices=8 --accelerator=cuda --precision="bf16"

Here is how to use `DeepSpeed Zero3 <https://www.deepspeed.ai/news/2021/03/07/zero3-offload.html>`_ with eight GPUs and mixed precision:

.. code-block:: bash

     lightning run model ./path/to/train.py --strategy=deepspeed --devices=8 --accelerator=cuda --precision=16

:class:`~pytorch_lightning.lite.LightningLite` can also figure it out automatically for you!

.. code-block:: bash

    lightning run model ./path/to/train.py --devices=auto --accelerator=auto --precision=16


You can also easily use distributed collectives if required.

.. code-block:: python

    lite = LightningLite()

    # Transfer and concatenate tensors across processes
    lite.all_gather(...)

    # Transfer an object from one process to all the others
    lite.broadcast(..., src=...)

    # The total number of processes running across all devices and nodes.
    lite.world_size

    # The global index of the current process across all devices and nodes.
    lite.global_rank

    # The index of the current process among the processes running on the local node.
    lite.local_rank

    # The index of the current node.
    lite.node_rank

    # Whether this global rank is rank zero.
    if lite.is_global_zero:
        # do something on rank 0
        ...

    # Wait for all processes to enter this call.
    lite.barrier()


The code stays agnostic, whether you are running on CPU, on two GPUS or on multiple machines with many GPUs.

If you require custom data or model device placement, you can deactivate :class:`~pytorch_lightning.lite.LightningLite`'s automatic placement by doing ``lite.setup_dataloaders(..., move_to_device=False)`` for the data and ``lite.setup(..., move_to_device=False)`` for the model.
Furthermore, you can access the current device from ``lite.device`` or rely on :meth:`~pytorch_lightning.lite.LightningLite.to_device` utility to move an object to the current device.


----------


Distributed Training Pitfalls
=============================

The :class:`~pytorch_lightning.lite.LightningLite` provides you with the tools to scale your training, but there are several major challenges ahead of you now:


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


Additionally, you can pass in your custom strategy by configuring additional parameters.

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

.. warning:: ``gpus=x`` has been deprecated in v1.7 and will be removed in v2.0.
    Please use ``accelerator='gpu'`` and ``devices=x`` instead.

Shorthand for setting ``devices=X`` and ``accelerator="gpu"``.

.. code-block:: python

    # Run on two GPUs
    lite = Lite(accelerator="gpu", devices=2)

    # Equivalent
    lite = Lite(devices=2, accelerator="gpu")


tpu_cores
=========

.. warning:: ``tpu_cores=x`` has been deprecated in v1.7 and will be removed in v2.0.
    Please use ``accelerator='tpu'`` and ``devices=x`` instead.

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


setup
=====

Set up a model and corresponding optimizer(s). If you need to set up multiple models, call ``setup()`` on each of them.
Moves the model and optimizer to the correct device automatically.

.. code-block:: python

    model = nn.Linear(32, 64)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Set up model and optimizer for accelerated training
    model, optimizer = lite.setup(model, optimizer)

    # If you don't want Lite to set the device
    model, optimizer = lite.setup(model, optimizer, move_to_device=False)


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

    train_data, test_data = lite.setup_dataloaders(train_data, test_data)

    # If you don't want Lite to move the data to the device
    train_data, test_data = lite.setup_dataloaders(train_data, test_data, move_to_device=False)

    # If you don't want Lite to replace the sampler in the context of distributed training
    train_data, test_data = lite.setup_dataloaders(train_data, test_data, replace_sampler=False)


backward
========

This replaces any occurrences of ``loss.backward()`` and makes your code accelerator and precision agnostic.

.. code-block:: python

    output = model(input)
    loss = loss_fn(output, target)

    # loss.backward()
    lite.backward(loss)


to_device
=========

Use :meth:`~pytorch_lightning.lite.lite.LightningLite.to_device` to move models, tensors or collections of tensors to
the current device. By default :meth:`~pytorch_lightning.lite.lite.LightningLite.setup` and
:meth:`~pytorch_lightning.lite.lite.LightningLite.setup_dataloaders` already move the model and data to the correct
device, so calling this method is only necessary for manual operation when needed.

.. code-block:: python

    data = torch.load("dataset.pt")
    data = lite.to_device(data)


seed_everything
===============

Make your code reproducible by calling this method at the beginning of your run.

.. code-block:: python

    # Instead of `torch.manual_seed(...)`, call:
    lite.seed_everything(1234)


This covers PyTorch, NumPy and Python random number generators. In addition, Lite takes care of properly initializing
the seed of dataloader worker processes (can be turned off by passing ``workers=False``).


autocast
========

Let the precision backend autocast the block of code under this context manager. This is optional and already done by
Lite for the model's forward method (once the model was :meth:`~pytorch_lightning.lite.lite.LightningLite.setup`).
You need this only if you wish to autocast more operations outside the ones in model forward:

.. code-block:: python

    model, optimizer = lite.setup(model, optimizer)

    # Lite handles precision automatically for the model
    output = model(inputs)

    with lite.autocast():  # optional
        loss = loss_function(output, target)

    lite.backward(loss)
    ...


print
=====

Print to the console via the built-in print function, but only on the main process.
This avoids excessive printing and logs when running on multiple devices/nodes.


.. code-block:: python

    # Print only on the main process
    lite.print(f"{epoch}/{num_epochs}| Train Epoch Loss: {loss}")


save
====

Save contents to a checkpoint. Replaces all occurrences of ``torch.save(...)`` in your code. Lite will take care of
handling the saving part correctly, no matter if you are running a single device, multi-devices or multi-nodes.

.. code-block:: python

    # Instead of `torch.save(...)`, call:
    lite.save(model.state_dict(), "path/to/checkpoint.ckpt")


load
====

Load checkpoint contents from a file. Replaces all occurrences of ``torch.load(...)`` in your code. Lite will take care of
handling the loading part correctly, no matter if you are running a single device, multi-device, or multi-node.

.. code-block:: python

    # Instead of `torch.load(...)`, call:
    lite.load("path/to/checkpoint.ckpt")


barrier
=======

Call this if you want all processes to wait and synchronize. Once all processes have entered this call,
execution continues. Useful for example when you want to download data on one process and make all others wait until
the data is written to disk.

.. code-block:: python

    # Download data only on one process
    if lite.global_rank == 0:
        download_data("http://...")

    # Wait until all processes meet up here
    lite.barrier()

    # All processes are allowed to read the data now


no_backward_sync
================

Use this context manager when performing gradient accumulation and using a distributed strategy (e.g., DDP).
It will speed up your training loop by cutting redundant communication between processes during the accumulation phase.

.. code-block:: python

    # Accumulate gradient 8 batches at a time
    is_accumulating = batch_idx % 8 != 0

    with lite.no_backward_sync(model, enabled=is_accumulating):
        output = model(input)
        loss = ...
        lite.backward(loss)
        ...

    # Step the optimizer every 8 batches
    if not is_accumulating:
        optimizer.step()
        optimizer.zero_grad()

Both the model's `.forward()` and the `lite.backward()` call need to run under this context as shown in the example above.
For single-device strategies, it is a no-op. There are strategies that don't support this:

- deepspeed
- dp
- xla

For these, the context manager falls back to a no-op and emits a warning.
