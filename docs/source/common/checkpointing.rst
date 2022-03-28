.. testsetup:: *

    import os
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _checkpointing:

#############
Checkpointing
#############

Lightning provides functions to save and load checkpoints.

Checkpointing your training allows you to resume a training process in case it was interrupted, fine-tune a model or use a pre-trained model for inference without having to retrain the model.



*******************
Checkpoint Contents
*******************

A Lightning checkpoint has everything needed to restore a training session including:

- 16-bit scaling factor (if using 16-bit precision training)
- Current epoch
- Global step
- LightningModule's state_dict
- State of all optimizers
- State of all learning rate schedulers
- State of all callbacks (for stateful callbacks)
- State of datamodule (for stateful datamodules)
- The hyperparameters used for that model if passed in as hparams (Argparse.Namespace)
- State of Loops (if using Fault-Tolerant training)


Individual Component States
===========================

Each component can save and load its state by implementing the PyTorch ``state_dict``, ``load_state_dict`` stateful protocol.
For details on implementing your own stateful callbacks and datamodules, refer to the individual docs pages at :doc:`callbacks <../extensions/callbacks>` and :doc:`datamodules <../extensions/datamodules>`.


Operating on Global Checkpoint Component States
===============================================

If you need to operate on the global component state (i.e. the entire checkpoint dictionary), you can read/add/delete/modify custom states in your checkpoints before they are being saved or loaded.
For this you can override :meth:`~pytorch_lightning.core.hooks.CheckpointHooks.on_save_checkpoint` and :meth:`~pytorch_lightning.core.hooks.CheckpointHooks.on_load_checkpoint` in your ``LightningModule``
or :meth:`~pytorch_lightning.callbacks.base.Callback.on_save_checkpoint` and :meth:`~pytorch_lightning.callbacks.base.Callback.on_load_checkpoint` methods in your ``Callback``.


*****************
Checkpoint Saving
*****************

Automatic Saving
================

Lightning automatically saves a checkpoint for you in your current working directory, with the state of your last training epoch. This makes sure you can resume training in case it was interrupted.

To change the checkpoint path pass in:

.. code-block:: python

    # saves checkpoints to '/your/path/to/save/checkpoints' at every epoch end
    trainer = Trainer(default_root_dir="/your/path/to/save/checkpoints")

You can retrieve the checkpoint after training by calling:

.. code-block:: python

        checkpoint_callback = ModelCheckpoint(dirpath="my/path/", save_top_k=2, monitor="val_loss")
        trainer = Trainer(callbacks=[checkpoint_callback])
        trainer.fit(model)
        checkpoint_callback.best_model_path


Disabling Checkpoints
=====================

You can disable checkpointing by passing:

.. testcode::

   trainer = Trainer(enable_checkpointing=False)


Manual Saving
=============

You can manually save checkpoints and restore your model from the checkpointed state using :meth:`~pytorch_lightning.trainer.trainer.Trainer.save_checkpoint`
and :meth:`~pytorch_lightning.core.saving.ModelIO.load_from_checkpoint`.

.. code-block:: python

    model = MyLightningModule(hparams)
    trainer.fit(model)
    trainer.save_checkpoint("example.ckpt")
    new_model = MyLightningModule.load_from_checkpoint(checkpoint_path="example.ckpt")


Manual Saving with Distributed Training Strategies
==================================================

Lightning also handles strategies where multiple processes are running, such as DDP. For example, when using the DDP strategy our training script is running across multiple devices at the same time.
Lightning automatically ensures that the model is saved only on the main process, whilst other processes do not interfere with saving checkpoints. This requires no code changes as seen below:

.. code-block:: python

    trainer = Trainer(strategy="ddp")
    model = MyLightningModule(hparams)
    trainer.fit(model)
    # Saves only on the main process
    trainer.save_checkpoint("example.ckpt")

Not using :meth:`~pytorch_lightning.trainer.trainer.Trainer.save_checkpoint` can lead to unexpected behavior and potential deadlock. Using other saving functions will result in all devices attempting to save the checkpoint. As a result, we highly recommend using the Trainer's save functionality.
If using custom saving functions cannot be avoided, we recommend using the :func:`~pytorch_lightning.utilities.rank_zero.rank_zero_only` decorator to ensure saving occurs only on the main process. Note that this will only work if all ranks hold the exact same state and won't work when using
model parallel distributed strategies such as deepspeed or sharded training.


Checkpointing Hyperparameters
=============================

The Lightning checkpoint also saves the arguments passed into the LightningModule init
under the ``"hyper_parameters"`` key in the checkpoint.

.. code-block:: python

    class MyLightningModule(LightningModule):
        def __init__(self, learning_rate, *args, **kwargs):
            super().__init__()
            self.save_hyperparameters()


    # all init args were saved to the checkpoint
    checkpoint = torch.load(CKPT_PATH)
    print(checkpoint["hyper_parameters"])
    # {"learning_rate": the_value}


-----------


******************
Checkpoint Loading
******************

To load a model along with its weights and hyperparameters use the following method:

.. code-block:: python

    model = MyLightningModule.load_from_checkpoint(PATH)

    print(model.learning_rate)
    # prints the learning_rate you used in this checkpoint

    model.eval()
    y_hat = model(x)

But if you don't want to use the hyperparameters saved in the checkpoint, pass in your own here:

.. testcode::

    class LitModel(LightningModule):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.save_hyperparameters()
            self.l1 = nn.Linear(self.hparams.in_dim, self.hparams.out_dim)

you can restore the model like this

.. code-block:: python

    # if you train and save the model like this it will use these values when loading
    # the weights. But you can overwrite this
    LitModel(in_dim=32, out_dim=10)

    # uses in_dim=32, out_dim=10
    model = LitModel.load_from_checkpoint(PATH)

    # uses in_dim=128, out_dim=10
    model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)


Restoring Training State
========================

If you don't just want to load weights, but instead restore the full training,
do the following:

.. code-block:: python

   model = LitModel()
   trainer = Trainer()

   # automatically restores model, epoch, step, LR schedulers, apex, etc...
   trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")


-----------


*******************************************
Conditional Checkpointing (ModelCheckpoint)
*******************************************

The :class:`~pytorch_lightning.callbacks.ModelCheckpoint` callback allows you to configure when/which/what/where checkpointing should happen. It follows the normal Callback hook structure so you can
hack it around/override its methods for your use-cases as well. Following are some of the common use-cases along with the arguments you need to specify to configure it:


How does it work?
=================

``ModelCheckpoint`` helps cover the following cases from WH-Family:

When
----

- When using iterative training which doesn't have an epoch, you can checkpoint at every ``N`` training steps by specifying ``every_n_training_steps=N``.
- You can also control the interval of epochs between checkpoints using ``every_n_epochs`` between checkpoints, to avoid slowdowns.
- You can checkpoint at a regular time interval using ``train_time_interval`` argument independent of the steps or epochs.
- In case you are monitoring a training metrics, we'd suggest using ``save_on_train_epoch_end=True`` to ensure the required metric is being accumulated correctly for creating a checkpoint.


Which
-----

- You can save the last checkpoint when training ends using ``save_last`` argument.

- You can save top-K and last-K checkpoints by configuring the ``monitor`` and ``save_top_k`` argument.

|

    .. testcode::

        from pytorch_lightning.callbacks import ModelCheckpoint


        # saves top-K checkpoints based on "val_loss" metric
        checkpoint_callback = ModelCheckpoint(
            save_top_k=10,
            monitor="val_loss",
            mode="min",
            dirpath="my/path/",
            filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        )

        # saves last-K checkpoints based on "global_step" metric
        # make sure you log it inside your LightningModule
        checkpoint_callback = ModelCheckpoint(
            save_top_k=10,
            monitor="global_step",
            mode="max",
            dirpath="my/path/",
            filename="sample-mnist-{epoch:02d}-{global_step}",
        )

-  You can customize the checkpointing behavior to monitor any quantity of your training or validation steps. For example, if you want to update your checkpoints based on your validation loss:

|

    .. testcode::

        from pytorch_lightning.callbacks import ModelCheckpoint


        class LitAutoEncoder(LightningModule):
            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self.backbone(x)

                # 1. calculate loss
                loss = F.cross_entropy(y_hat, y)

                # 2. log val_loss
                self.log("val_loss", loss)


        # 3. Init ModelCheckpoint callback, monitoring "val_loss"
        checkpoint_callback = ModelCheckpoint(monitor="val_loss")

        # 4. Add your callback to the callbacks list
        trainer = Trainer(callbacks=[checkpoint_callback])


What
----

- By default, the ``ModelCheckpoint`` callback saves model weights, optimizer states, etc., but in case you have limited disk space or just need the model weights to be saved you can specify ``save_weights_only=True``.


Where
-----

- It gives you the ability to specify the ``dirpath`` and ``filename`` for your checkpoints. Filename can also be dynamic so you can inject the metrics that are being logged using :meth:`~pytorch_lightning.core.lightning.LightningModule.log`.

|

    .. testcode::

        from pytorch_lightning.callbacks import ModelCheckpoint


        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint_callback = ModelCheckpoint(
            dirpath="my/path/",
            filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        )

|

The :class:`~pytorch_lightning.callbacks.ModelCheckpoint` callback is very robust and should cover 99% of the use-cases. If you find a use-case that is not configured yet, feel free to open an issue with a feature request on GitHub
and the Lightning Team will be happy to integrate/help integrate it.


-----------


***********************
Customize Checkpointing
***********************

.. warning::

    The Checkpoint IO API is experimental and subject to change.


Lightning supports modifying the checkpointing save/load functionality through the ``CheckpointIO``. This encapsulates the save/load logic
that is managed by the ``Strategy``. ``CheckpointIO`` is different from :meth:`~pytorch_lightning.core.hooks.CheckpointHooks.on_save_checkpoint`
and :meth:`~pytorch_lightning.core.hooks.CheckpointHooks.on_load_checkpoint` methods as it determines how the checkpoint is saved/loaded to storage rather than
what's saved in the checkpoint.


Built-in Checkpoint IO Plugins
==============================

.. list-table:: Built-in Checkpoint IO Plugins
   :widths: 25 75
   :header-rows: 1

   * - Plugin
     - Description
   * - :class:`~pytorch_lightning.plugins.io.TorchCheckpointIO`
     - CheckpointIO that utilizes :func:`torch.save` and :func:`torch.load` to save and load checkpoints
       respectively, common for most use cases.
   * - :class:`~pytorch_lightning.plugins.io.XLACheckpointIO`
     - CheckpointIO that utilizes :func:`xm.save` to save checkpoints for TPU training strategies.


Custom Checkpoint IO Plugin
===========================

``CheckpointIO`` can be extended to include your custom save/load functionality to and from a path. The ``CheckpointIO`` object can be passed to either a ``Trainer`` directly or a ``Strategy`` as shown below:

.. code-block:: python

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.plugins import CheckpointIO
    from pytorch_lightning.strategies import SingleDeviceStrategy


    class CustomCheckpointIO(CheckpointIO):
        def save_checkpoint(self, checkpoint, path, storage_options=None):
            ...

        def load_checkpoint(self, path, storage_options=None):
            ...

        def remove_checkpoint(self, path):
            ...


    custom_checkpoint_io = CustomCheckpointIO()

    # Either pass into the Trainer object
    model = MyModel()
    trainer = Trainer(
        plugins=[custom_checkpoint_io],
        callbacks=ModelCheckpoint(save_last=True),
    )
    trainer.fit(model)

    # or pass into Strategy
    model = MyModel()
    device = torch.device("cpu")
    trainer = Trainer(
        strategy=SingleDeviceStrategy(device, checkpoint_io=custom_checkpoint_io),
        callbacks=ModelCheckpoint(save_last=True),
    )
    trainer.fit(model)

.. note::

    Some ``TrainingTypePlugins`` like ``DeepSpeedStrategy`` do not support custom ``CheckpointIO`` as checkpointing logic is not modifiable.

-----------

***************************
Managing Remote Filesystems
***************************

Lightning supports saving and loading checkpoints from a variety of filesystems, including local filesystems and several cloud storage providers.

Check out :ref:`Remote Filesystems <remote_fs>` document for more info.
