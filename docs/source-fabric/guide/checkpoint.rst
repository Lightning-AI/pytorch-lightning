##############################
Saving and Loading Checkpoints
##############################

Fabric makes it easy and efficient to save the state of your training loop into a checkpoint file, no matter how large your model is.

----


********************************
Define the state of your program
********************************

To save and resume your training, you need to define which variables in your program you want to have saved.
Put everything into a dictionary, including models and optimizers and whatever metadata you have:

.. code-block:: python

    # Define the state of your program/loop
    state = {"model1": model1, "model2": model2, "optimizer": optimizer, "iteration": iteration, "hparams": ...}


----


*****************
Save a checkpoint
*****************

To save the state to the filesystem, pass it to the :meth:`~lightning.fabric.fabric.Fabric.save` method:

.. code-block:: python

    fabric.save("path/to/checkpoint.ckpt", state)

This will unwrap your model and optimizer and automatically convert their ``state_dict`` for you.
Fabric and the underlying strategy will decide in which format your checkpoint gets saved.
For example, ``strategy="ddp"`` saves a single file on rank 0, while ``strategy="fsdp"`` saves multiple files from all ranks.


----


*************************
Restore from a checkpoint
*************************

You can restore the state by loading a saved checkpoint back with :meth:`~lightning.fabric.fabric.Fabric.load`:

.. code-block:: python

    fabric.load("path/to/checkpoint.ckpt", state)

Fabric will replace the state of your objects in-place.
You can also request only to restore a portion of the checkpoint.
For example, you want only to restore the model weights in your inference script:

.. code-block:: python

    state = {"model1": model1}
    remainder = fabric.load("path/to/checkpoint.ckpt", state)

The remainder of the checkpoint that wasn't restored gets returned in case you want to do something else with it.
If you want to be in complete control of how states get restored, you can omit passing a state and get the entire raw checkpoint dictionary returned:

.. code-block:: python

    # Request the raw checkpoint
    full_checkpoint = fabric.load("path/to/checkpoint.ckpt")

    model.load_state_dict(full_checkpoint["model"])
    optimizer.load_state_dict(full_checkpoint["optimizer"])
    ...



----


*************************
Load a partial checkpoint
*************************

Loading a checkpoint is normally "strict", meaning parameter names in the checkpoint must match the parameter names in the model.
However, when loading checkpoints for fine-tuning or transfer learning, it can happen that only a portion of the parameters match the model.
For this case, you can disable strict loading to avoid errors:

.. code-block:: python

    state = {"model": model}

    # strict loading is the default
    fabric.load("path/to/checkpoint.ckpt", state, strict=True)

    # disable strict loading
    fabric.load("path/to/checkpoint.ckpt", state, strict=False)


Here is a trivial example to illustrate how it works:

.. code-block:: python

    import torch
    import lightning as L

    fabric = L.Fabric()

    # Save a checkpoint of a trained model
    model1 = torch.nn.Linear(2, 2, bias=True)
    state = {"model": model1}
    fabric.save("state.ckpt", state)

    # Later on, make a new model that misses a parameter
    model2 = torch.nn.Linear(2, 2, bias=False)
    state = {"model": model2}

    # `strict=True` would lead to an error, because the bias
    # parameter is missing, but we can load the rest of the
    # parameters successfully
    fabric.load("state.ckpt", state, strict=False)


See also: `Saving and loading models in PyTorch <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_.


----

*************************
Save a partial checkpoint
*************************

When saving a checkpoint using Fabric, you have the flexibility to choose which parameters to include in the saved file.
This can be useful in scenarios such as fine-tuning, where you only want to save a subset of the parameters, reducing
the size of the checkpoint and saving disk space.

To accomplish this, you can use filters during the saving process. The filter is a function that determines whether
an item should be saved (returning ``True``) or excluded (returning ``False``).
The filter operates on dictionary objects and evaluates each key-value pair individually.

Here's an example of using a filter when saving a checkpoint:

.. code-block:: python

    state = {"model": model, "optimizer": optimizer, "foo": 123}

    # save only the model weights
    filter = {"model": lambda k, v: "weight"}
    fabric.save("path/to/checkpoint.ckpt", state, filter=filter)
    # This will save {"model": {"layer.weight": ...}, "optimizer": ..., "foo": 123}
    # note that the optimizer params corresponding to the excluded model params are not filtered


----


**********
Next steps
**********

Learn from our template how Fabrics checkpoint mechanism can be integrated into a full Trainer:

.. raw:: html

    <div class="display-card-container">
        <div class="row">

.. displayitem::
    :header: Trainer Template
    :description: Take our Fabric Trainer template and customize it for your needs
    :button_link: https://github.com/Lightning-AI/lightning/tree/master/examples/fabric/build_your_own_trainer
    :col_css: col-md-4
    :height: 150
    :tag: intermediate

.. raw:: html

        </div>
    </div>
