:orphan:

.. _checkpointing_basic:

######################################
Saving and loading checkpoints (basic)
######################################
**Audience:** All users

----

*********************
What is a checkpoint?
*********************
When a model is training, the performance changes as it continues to see more data. It is a best practice to save the state of a model throughout the training process. This gives you a version of the model, *a checkpoint*, at each key point during the development of the model. Once training has completed, use the checkpoint that corresponds to the best performance you found during the training process.

Checkpoints also enable your training to resume from where it was in case the training process is interrupted.

PyTorch Lightning checkpoints are fully usable in plain PyTorch.

----

************************
Contents of a checkpoint
************************
A Lightning checkpoint contains a dump of the model's entire internal state. Unlike plain PyTorch, Lightning saves *everything* you need to restore a model even in the most complex distributed training environments.

Inside a Lightning checkpoint you'll find:

- 16-bit scaling factor (if using 16-bit precision training)
- Current epoch
- Global step
- LightningModule's state_dict
- State of all optimizers
- State of all learning rate schedulers
- State of all callbacks (for stateful callbacks)
- State of datamodule (for stateful datamodules)
- The hyperparameters (init arguments) with which the model was created
- The hyperparameters (init arguments) with which the datamodule was created
- State of Loops

----

*****************
Save a checkpoint
*****************
Lightning automatically saves a checkpoint for you in your current working directory, with the state of your last training epoch. This makes sure you can resume training in case it was interrupted.

.. code-block:: python

    # simply by using the Trainer you get automatic checkpointing
    trainer = Trainer()

To change the checkpoint path use the `default_root_dir` argument:

.. code-block:: python

    # saves checkpoints to 'some/path/' at every epoch end
    trainer = Trainer(default_root_dir="some/path/")


----


*******************************
LightningModule from checkpoint
*******************************

To load a LightningModule along with its weights and hyperparameters use the following method:

.. code-block:: python

    model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")

    # disable randomness, dropout, etc...
    model.eval()

    # predict with the model
    y_hat = model(x)

----

Save hyperparameters
====================
The LightningModule allows you to automatically save all the hyperparameters passed to *init* simply by calling *self.save_hyperparameters()*.

.. code-block:: python

    class MyLightningModule(LightningModule):
        def __init__(self, learning_rate, another_parameter, *args, **kwargs):
            super().__init__()
            self.save_hyperparameters()

The hyperparameters are saved to the "hyper_parameters" key in the checkpoint

.. code-block:: python

    checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    print(checkpoint["hyper_parameters"])
    # {"learning_rate": the_value, "another_parameter": the_other_value}

The LightningModule also has access to the Hyperparameters

.. code-block:: python

    model = MyLightningModule.load_from_checkpoint("/path/to/checkpoint.ckpt")
    print(model.learning_rate)

----

Initialize with other parameters
================================
If you used the *self.save_hyperparameters()* method in the *__init__* method of the LightningModule, you can override these and initialize the model with different hyperparameters.

.. code-block:: python

    # if you train and save the model like this it will use these values when loading
    # the weights. But you can overwrite this
    LitModel(in_dim=32, out_dim=10)

    # uses in_dim=32, out_dim=10
    model = LitModel.load_from_checkpoint(PATH)

    # uses in_dim=128, out_dim=10
    model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)

In some cases, we may also pass entire PyTorch modules to the ``__init__`` method, which you don't want to save as hyperparameters due to their large size. If you didn't call ``self.save_hyperparameters()`` or ignore parameters via ``save_hyperparameters(ignore=...)``, then you must pass the missing positional arguments or keyword arguments when calling ``load_from_checkpoint`` method:


.. code-block:: python

    class LitAutoencoder(L.LightningModule):
        def __init__(self, encoder, decoder):
            ...

        ...


    model = LitAutoEncoder.load_from_checkpoint(PATH, encoder=encoder, decoder=decoder)


----


*************************
nn.Module from checkpoint
*************************
Lightning checkpoints are fully compatible with plain torch nn.Modules.

.. code-block:: python

    checkpoint = torch.load(CKPT_PATH)
    print(checkpoint.keys())

For example, let's pretend we created a LightningModule like so:

.. code-block:: python

    class Encoder(nn.Module):
        ...


    class Decoder(nn.Module):
        ...


    class Autoencoder(L.LightningModule):
        def __init__(self, encoder, decoder, *args, **kwargs):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder


    autoencoder = Autoencoder(Encoder(), Decoder())

Once the autoencoder has trained, pull out the relevant weights for your torch nn.Module:

.. code-block:: python

    checkpoint = torch.load(CKPT_PATH)
    encoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("encoder.")}
    decoder_weights = {k: v for k, v in checkpoint["state_dict"].items() if k.startswith("decoder.")}


----


*********************
Disable checkpointing
*********************

You can disable checkpointing by passing:

.. testcode::

   trainer = Trainer(enable_checkpointing=False)

----

*********************
Resume training state
*********************

If you don't just want to load weights, but instead restore the full training, do the following:

.. code-block:: python

   model = LitModel()
   trainer = Trainer()

   # automatically restores model, epoch, step, LR schedulers, etc...
   trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
