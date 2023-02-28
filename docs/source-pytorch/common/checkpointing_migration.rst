:orphan:

.. _checkpointing_intermediate_2:

####################################
Upgrading checkpoints (intermediate)
####################################
**Audience:** Users who are upgrading Lightning and their code and want to reuse their old checkpoints.

----

**************************************
Resume training from an old checkpoint
**************************************

Next to the model weights and trainer state, a Lightning checkpoint contains the version number of Lightning with which the checkpoint was saved.
When you load a checkpoint file, either by resuming training

.. code-block:: python

    trainer = Trainer(...)
    trainer.fit(model, ckpt_path="path/to/checkpoint.ckpt")

or by loading the state directly into your model,

.. code-block:: python

    model = LitModel.load_from_checkpoint("path/to/checkpoint.ckpt")

Lightning will automatically recognize that it is from an older version and migrates the internal structure so it can be loaded properly.
This is done without any action required by the user.

----

************************************
Upgrade checkpoint files permanently
************************************

When Lightning loads a checkpoint, it applies the version migration on-the-fly as explained above, but it does not modify your checkpoint files.
You can upgrade checkpoint files permanently with the following command

.. code-block::

    python -m lightning.pytorch.utilities.upgrade_checkpoint path/to/model.ckpt


or a folder with multiple files:

.. code-block::

    python -m lightning.pytorch.utilities.upgrade_checkpoint /path/to/checkpoints/folder
