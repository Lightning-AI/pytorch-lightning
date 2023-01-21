:orphan:

##############################
Saving and Loading Checkpoints
##############################

Fabric makes it easy and efficient to save the state of your training loop into a checkpoint file, no matter how large your model is.

----


********************************
Define the state of your program
********************************

In order save and resume your training, you need to define which variables in your program you want to have saved.
Put everything into a dictionary, including models and optimizers and whatever metadata you have:

.. code-block:: python

    # Define the state of your program/loop
    state = {"model1": model1, "model2": model2, "optimizer": optimizer, "iteration": iteration, "hparams": ...}


----


*****************
Save a checkpoint
*****************

To save the state to the filesystem, pass it to the :meth:`~lightning_fabric.fabric.Fabric.save` method:

.. code-block:: python

    fabric.save("path/to/checkpoint.ckpt", state)

This will unwrap your model and optimizer and convert their `state_dict` automaticall for you.
Fabric and the underlying strategy will decide in which format your checkpoint gets saved.
For example, ``strategy="ddp"`` saves a single file on rank 0, while ``strategy="fsdp"`` saves multiple files from all ranks.


----


*************************
Restore from a checkpoint
*************************

You can restore the state by loading a saved checkpoint back with :meth:`~lightning_fabric.fabric.Fabric.load`:

.. code-block:: python

    fabric.load("path/to/checkpoint.ckpt", state)

Fabric will replace the state of your objects in-place.
You can also request to only restore a portion of the checkpoint.
For example, you want to only restore the model weights in your inference script:

.. code-block:: python

    state = {"model1": model1}
    remainder = fabric.load("path/to/checkpoint.ckpt", state)

The remainder of the checkpoint that wasn't restored gets returned in case you want to do something else with it.
If you want to be in full control of how states get restored, you can omit passing a state and get the full raw checkpoint dictionary returned:

.. code-block:: python

    # Request the raw checkpoint
    full_checkpoint = fabric.load("path/to/checkpoint.ckpt")

    model.load_state_dict(full_checkpoint["model"])
    optimizer.load_state_dict(full_checkpoint["optimizer"])
    ...
