##########################################
Saving and Loading Distributed Checkpoints
##########################################

Generally, the bigger your model is, the longer it takes to save a checkpoint to disk.
With distributed checkpoints (sometimes called sharded checkpoints), you can save and load the state of your training script with multiple GPUs or nodes more efficiently, avoiding memory issues.


----


*****************************
Save a distributed checkpoint
*****************************

The distributed checkpoint format is the default when you train with the :doc:`FSDP strategy <../../advanced/model_parallel/fsdp>`.

.. code-block:: python

    import lightning as L
    from lightning.fabric.strategies import FSDPStrategy

    # 1. Select the FSDP strategy
    strategy = FSDPStrategy(
        # Default: sharded/distributed checkpoint
        state_dict_type="sharded",
        # Full checkpoint (not distributed)
        # state_dict_type="full",
    )

    fabric = L.Fabric(devices=2, strategy=strategy, ...)
    fabric.launch()
    ...
    model, optimizer = fabric.setup(model, optimizer)

    # 2. Define model, optimizer, and other training loop state
    state = {"model": model, "optimizer": optimizer, "iter": iteration}

    # DON'T do this (inefficient):
    # state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), ...}

    # 3. Save using Fabric's method
    fabric.save("path/to/checkpoint/file", state)

    # DON'T do this (inefficient):
    # torch.save("path/to/checkpoint/file", state)

With ``state_dict_type="sharded"``, each process/GPU will save its own file into a folder at the given path.
This reduces memory peaks and speeds up the saving to disk.

.. collapse:: Full example

    .. code-block:: python

        import time
        import torch
        import torch.nn.functional as F

        import lightning as L
        from lightning.fabric.strategies import FSDPStrategy
        from lightning.pytorch.demos import Transformer, WikiText2

        strategy = FSDPStrategy(state_dict_type="sharded")
        fabric = L.Fabric(accelerator="cuda", devices=4, strategy=strategy)
        fabric.launch()

        with fabric.rank_zero_first():
            dataset = WikiText2()

        # 1B parameters
        model = Transformer(vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        model, optimizer = fabric.setup(model, optimizer)

        state = {"model": model, "optimizer": optimizer, "iteration": 0}

        for i in range(10):
            input, target = fabric.to_device(dataset[i])
            output = model(input.unsqueeze(0), target.unsqueeze(0))
            loss = F.nll_loss(output, target.view(-1))
            fabric.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            fabric.print(loss.item())

        fabric.print("Saving checkpoint ...")
        t0 = time.time()
        fabric.save("my-checkpoint.ckpt", state)
        fabric.print(f"Took {time.time() - t0:.2f} seconds.")

    Check the contents of the checkpoint folder:

    .. code-block:: bash

        ls -a my-checkpoint.ckpt/

    .. code-block::

        my-checkpoint.ckpt/
        ├── __0_0.distcp
        ├── __1_0.distcp
        ├── __2_0.distcp
        ├── __3_0.distcp
        ├── .metadata
        └── meta.pt

    The ``.distcp`` files contain the tensor shards from each process/GPU. You can see that the size of these files
    is roughly 1/4 of the total size of the checkpoint since the script distributes the model across 4 GPUs.


----


*****************************
Load a distributed checkpoint
*****************************

You can easily load a distributed checkpoint in Fabric if your script uses :doc:`FSDP <../../advanced/model_parallel/fsdp>`.

.. code-block:: python

    import lightning as L
    from lightning.fabric.strategies import FSDPStrategy

    # 1. Select the FSDP strategy
    fabric = L.Fabric(devices=2, strategy=FSDPStrategy(), ...)
    fabric.launch()
    ...
    model, optimizer = fabric.setup(model, optimizer)

    # 2. Define model, optimizer, and other training loop state
    state = {"model": model, "optimizer": optimizer, "iter": iteration}

    # 3. Load using Fabric's method
    fabric.load("path/to/checkpoint/file", state)

    # DON'T do this (inefficient):
    # model.load_state_dict(torch.load("path/to/checkpoint/file"))

Note that you can load the distributed checkpoint even if the world size has changed, i.e., you are running on a different number of GPUs than when you saved the checkpoint.

.. collapse:: Full example

    .. code-block:: python

        import torch

        import lightning as L
        from lightning.fabric.strategies import FSDPStrategy
        from lightning.pytorch.demos import Transformer, WikiText2

        strategy = FSDPStrategy(state_dict_type="sharded")
        fabric = L.Fabric(accelerator="cuda", devices=2, strategy=strategy)
        fabric.launch()

        with fabric.rank_zero_first():
            dataset = WikiText2()

        # 1B parameters
        model = Transformer(vocab_size=dataset.vocab_size, nlayers=32, nhid=4096, ninp=1024, nhead=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

        model, optimizer = fabric.setup(model, optimizer)

        state = {"model": model, "optimizer": optimizer, "iteration": 0}

        fabric.print("Loading checkpoint ...")
        fabric.load("my-checkpoint.ckpt", state)


.. important::

    If you want to load a distributed checkpoint into a script that doesn't use FSDP (or Fabric at all), then you will have to :ref:`convert it to a single-file checkpoint first <Convert dist-checkpoint>`.


----


.. _Convert dist-checkpoint:

********************************
Convert a distributed checkpoint
********************************

It is possible to convert a distributed checkpoint to a regular, single-file checkpoint with this utility:

.. code-block:: bash

    fabric consolidate path/to/my/checkpoint

You will need to do this for example if you want to load the checkpoint into a script that doesn't use FSDP, or need to export the checkpoint to a different format for deployment, evaluation, etc.

.. note::

    All tensors in the checkpoint will be converted to CPU tensors, and no GPUs are required to run the conversion command.
    This function assumes you have enough free CPU memory to hold the entire checkpoint in memory.

.. collapse:: Full example

    Assuming you have saved a checkpoint ``my-checkpoint.ckpt`` using the examples above, run the following command to convert it:

    .. code-block:: bash

        fabric consolidate my-checkpoint.ckpt

    This saves a new file ``my-checkpoint.ckpt.consolidated`` next to the sharded checkpoint which you can load normally in PyTorch:

    .. code-block:: python

        import torch

        checkpoint = torch.load("my-checkpoint.ckpt.consolidated")
        print(list(checkpoint.keys()))
        print(checkpoint["model"]["transformer.decoder.layers.31.norm1.weight"])


|
