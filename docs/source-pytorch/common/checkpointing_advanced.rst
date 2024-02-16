.. _checkpointing_advanced:

##################################
Cloud-based checkpoints (advanced)
##################################


*****************
Cloud checkpoints
*****************
Lightning is integrated with the major remote file systems including local filesystems and several cloud storage providers such as
`S3 <https://aws.amazon.com/s3/>`_ on `AWS <https://aws.amazon.com/>`_, `GCS <https://cloud.google.com/storage>`_ on `Google Cloud <https://cloud.google.com/>`_,
or `ADL <https://azure.microsoft.com/solutions/data-lake/>`_ on `Azure <https://azure.microsoft.com/>`_.

PyTorch Lightning uses `fsspec <https://filesystem-spec.readthedocs.io/>`_ internally to handle all filesystem operations.

----

Save a cloud checkpoint
=======================

To save to a remote filesystem, prepend a protocol like "s3:/" to the root_dir used for writing and reading model data.

.. code-block:: python

    # `default_root_dir` is the default path used for logs and checkpoints
    trainer = Trainer(default_root_dir="s3://my_bucket/data/")
    trainer.fit(model)

----

Resume training from a cloud checkpoint
=======================================
To resume training from a cloud checkpoint use a cloud url.

.. code-block:: python

    trainer = Trainer(default_root_dir=tmpdir, max_steps=3)
    trainer.fit(model, ckpt_path="s3://my_bucket/ckpts/classifier.ckpt")

PyTorch Lightning uses `fsspec <https://filesystem-spec.readthedocs.io/>`_ internally to handle all filesystem operations.

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
