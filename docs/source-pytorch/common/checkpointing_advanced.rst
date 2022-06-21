.. _checkpointing_advanced:

########################
Checkpointing (advanced)
########################


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
When you need to change the components of a checkpoint before saving or loading, use the :meth:`~pytorch_lightning.core.hooks.CheckpointHooks.on_save_checkpoint` and :meth:`~pytorch_lightning.core.hooks.CheckpointHooks.on_load_checkpoint` of your ``LightningModule``.

.. code:: python

    class LitModel(pl.LightningModule):
        def on_save_checkpoint(self, checkpoint):
            checkpoint["something_cool_i_want_to_save"] = my_cool_pickable_object

        def on_load_checkpoint(self, checkpoint):
            my_cool_pickable_object = checkpoint["something_cool_i_want_to_save"]

Use the above approach when you need to couple this behavior to your LightningModule for reproducibility reasons. Otherwise, Callbacks also have the :meth:`~pytorch_lightning.callbacks.callback.Callback.on_save_checkpoint` and :meth:`~pytorch_lightning.callbacks.callback.Callback.on_load_checkpoint` which you should use instead:

.. code:: python

    class LitCallback(pl.Callback):
        def on_save_checkpoint(self, checkpoint):
            checkpoint["something_cool_i_want_to_save"] = my_cool_pickable_object

        def on_load_checkpoint(self, checkpoint):
            my_cool_pickable_object = checkpoint["something_cool_i_want_to_save"]
