:orphan:

.. _checkpointing_intermediate_1:

###############################################
Customize checkpointing behavior (intermediate)
###############################################
**Audience:** Users looking to customize the checkpointing behavior

----

*****************************
Modify checkpointing behavior
*****************************
For fine-grained control over checkpointing behavior, use the :class:`~lightning.pytorch.callbacks.ModelCheckpoint` object

.. code-block:: python

        from lightning.pytorch.callbacks import ModelCheckpoint

        checkpoint_callback = ModelCheckpoint(dirpath="my/path/", save_top_k=2, monitor="val_loss")
        trainer = Trainer(callbacks=[checkpoint_callback])
        trainer.fit(model)
        checkpoint_callback.best_model_path

Any value that has been logged via *self.log* in the LightningModule can be monitored.

.. code-block:: python

        class LitModel(L.LightningModule):
            def training_step(self, batch, batch_idx):
                self.log("my_metric", x)


        # 'my_metric' is now able to be monitored
        checkpoint_callback = ModelCheckpoint(monitor="my_metric")

----

*****************************
Save checkpoints by condition
*****************************
To save checkpoints based on a (*when/which/what/where*) condition (for example *when* the validation_loss is lower) modify the :class:`~lightning.pytorch.callbacks.ModelCheckpoint` properties.

When
====

- When using iterative training which doesn't have an epoch, you can checkpoint at every ``N`` training steps by specifying ``every_n_train_steps=N``.
- You can also control the interval of epochs between checkpoints using ``every_n_epochs``, to avoid slowdowns.
- You can checkpoint at a regular time interval using the ``train_time_interval`` argument independent of the steps or epochs.
- In case you are monitoring a training metric, we'd suggest using ``save_on_train_epoch_end=True`` to ensure the required metric is being accumulated correctly for creating a checkpoint.


Which
=====

- You can save the last checkpoint when training ends using ``save_last`` argument.
- You can save top-K and last-K checkpoints by configuring the ``monitor`` and ``save_top_k`` argument.

|

    .. testcode::

        from lightning.pytorch.callbacks import ModelCheckpoint


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

    .. note::

        It is recommended that you pass formatting options to ``filename`` to include the monitored metric like shown
        in the example above. Otherwise, if ``save_top_k >= 2`` and ``enable_version_counter=True`` (default), a
        version is appended to the ``filename`` to prevent filename collisions. You should not rely on the appended
        version to retrieve the top-k model, since there is no relationship between version count and model performance.
        For example, ``filename-v2.ckpt`` doesn't necessarily correspond to the top-2 model.


-  You can customize the checkpointing behavior to monitor any quantity of your training or validation steps. For example, if you want to update your checkpoints based on your validation loss:

|

    .. testcode::

        from lightning.pytorch.callbacks import ModelCheckpoint


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
====

- By default, the ``ModelCheckpoint`` callback saves model weights, optimizer states, etc., but in case you have limited disk space or just need the model weights to be saved you can specify ``save_weights_only=True``.


Where
=====

- By default, the ``ModelCheckpoint`` will save files into the ``Trainer.log_dir``. It gives you the ability to specify the ``dirpath`` and ``filename`` for your checkpoints. Filename can also be dynamic so you can inject the metrics that are being logged using :meth:`~lightning.pytorch.core.LightningModule.log`.

|

    .. testcode::

        from lightning.pytorch.callbacks import ModelCheckpoint


        # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
        checkpoint_callback = ModelCheckpoint(
            dirpath="my/path/",
            filename="sample-mnist-{epoch:02d}-{val_loss:.2f}",
        )

|

The :class:`~lightning.pytorch.callbacks.ModelCheckpoint` callback is very robust and should cover 99% of the use-cases. If you find a use-case that is not configured yet, feel free to open an issue with a feature request on GitHub
and the Lightning Team will be happy to integrate/help integrate it.

----

*************************
Save checkpoints manually
*************************

You can manually save checkpoints and restore your model from the checkpointed state using :meth:`~lightning.pytorch.trainer.trainer.Trainer.save_checkpoint`
and :meth:`~lightning.pytorch.core.LightningModule.load_from_checkpoint`.

.. code-block:: python

    model = MyLightningModule(hparams)
    trainer.fit(model)
    trainer.save_checkpoint("example.ckpt")

    # load the checkpoint later as normal
    new_model = MyLightningModule.load_from_checkpoint(checkpoint_path="example.ckpt")

Manual saving with distributed training
=======================================
In distributed training cases where a model is running across many machines, Lightning ensures that only one checkpoint is saved instead of a model per machine. This requires no code changes as seen below:

.. code-block:: python

    trainer = Trainer(strategy="ddp")
    model = MyLightningModule(hparams)
    trainer.fit(model)

    # Saves only on the main process
    # Handles strategy-specific saving logic like XLA, FSDP, DeepSpeed etc.
    trainer.save_checkpoint("example.ckpt")


By using :meth:`~lightning.pytorch.trainer.trainer.Trainer.save_checkpoint` instead of ``torch.save``, you make your code agnostic to the distributed training strategy being used.
It will ensure that checkpoints are saved correctly in a multi-process setting, avoiding race conditions, deadlocks and other common issues that normally require boilerplate code to handle properly.


----


***************************
Modularize your checkpoints
***************************
Checkpoints can also save the state of :doc:`datamodules <../extensions/datamodules_state>` and :doc:`callbacks <../extensions/callbacks_state>`.


----


****************************
Modify a checkpoint anywhere
****************************
When you need to change the components of a checkpoint before saving or loading, use the :meth:`~lightning.pytorch.core.hooks.CheckpointHooks.on_save_checkpoint` and :meth:`~lightning.pytorch.core.hooks.CheckpointHooks.on_load_checkpoint` of your ``LightningModule``.

.. code-block:: python

    class LitModel(L.LightningModule):
        def on_save_checkpoint(self, checkpoint):
            checkpoint["something_cool_i_want_to_save"] = my_cool_pickable_object

        def on_load_checkpoint(self, checkpoint):
            my_cool_pickable_object = checkpoint["something_cool_i_want_to_save"]

Use the above approach when you need to couple this behavior to your LightningModule for reproducibility reasons. Otherwise, Callbacks also have the :meth:`~lightning.pytorch.callbacks.callback.Callback.on_save_checkpoint` and :meth:`~lightning.pytorch.callbacks.callback.Callback.on_load_checkpoint` which you should use instead:

.. code-block:: python

    import lightning as L


    class LitCallback(L.Callback):
        def on_save_checkpoint(self, checkpoint):
            checkpoint["something_cool_i_want_to_save"] = my_cool_pickable_object

        def on_load_checkpoint(self, checkpoint):
            my_cool_pickable_object = checkpoint["something_cool_i_want_to_save"]


----


********************************
Resume from a partial checkpoint
********************************

Loading a checkpoint is normally "strict", meaning parameter names in the checkpoint must match the parameter names in the model or otherwise PyTorch will raise an error.
In use cases where you want to load only a partial checkpoint, you can disable strict loading by setting ``self.strict_loading = False`` in the LightningModule to avoid errors.
A common use case is when you have a pretrained feature extractor or encoder that you don't update during training, and you don't want it included in the checkpoint:

.. code-block:: python

    import lightning as L

    class LitModel(L.LightningModule):
        def __init__(self):
            super().__init__()

            # This model only trains the decoder, we don't save the encoder
            self.encoder = from_pretrained(...).requires_grad_(False)
            self.decoder = Decoder()

            # Set to False because we only care about the decoder
            self.strict_loading = False

        def state_dict(self):
            # Don't save the encoder, it is not being trained
            return {k: v for k, v in super().state_dict().items() if "encoder" not in k}


Since ``strict_loading`` is set to ``False``, you won't get any key errors when resuming the checkpoint with the Trainer:

.. code-block:: python

    trainer = Trainer()
    model = LitModel()

    # Will load weights with `.load_state_dict(strict=model.strict_loading)`
    trainer.fit(model, ckpt_path="path/to/checkpoint")
