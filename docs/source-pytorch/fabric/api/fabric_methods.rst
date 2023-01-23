:orphan:

##############
Fabric Methods
##############


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

See also: :doc:`../fundamentals/precision`


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


call
====

Use this to run all registered callback hooks with a given name and inputs.
It is useful when building a Trainer that allows the user to run arbitrary code at fixed points in the training loop.

.. code-block:: python

    class MyCallback:
        def on_train_start(self):
            ...

        def on_train_epoch_end(self, model, results):
            ...


    fabric = Fabric(callbacks=[MyCallback()])

    # Call any hook by name
    fabric.call("on_train_start")

    # Pass in additional arguments that the hook requires
    fabric.call("on_train_epoch_end", model=..., results={...})

    # Only the callbacks that have this method defined will be executed
    fabric.call("undefined")


See also: :doc:`../guide/callbacks`


log and log_dict
================

These methods allows you to send scalar metrics to a logger registered in Fabric.

.. code-block:: python

    # Set the logger in Fabric
    fabric = Fabric(loggers=TensorBoardLogger(...))

    # Anywhere in your training loop or model:
    fabric.log("loss", loss)

    # Or send multiple metrics at once:
    fabric.log_dict({"loss": loss, "accuracy": acc})

If no loggers are given to Fabric (default), ``log`` and ``log_dict`` won't do anything.
Here is what's happening under the hood (pseudo code) when you call ``.log()`` or ``log_dict``:

.. code-block:: python

    # When you call .log() or .log_dict(), we do this:
    for logger in fabric.loggers:
        logger.log_metrics(metrics=metrics, step=step)

See also: :doc:`../guide/logging`
