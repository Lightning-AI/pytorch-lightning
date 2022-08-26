:orphan:

.. _checkpointing_expert:

######################
Checkpointing (expert)
######################

*********************************
Writing your own Checkpoint class
*********************************

We provide ``Checkpoint`` class, for easier subclassing. Users may want to subclass this class in case of writing custom ``ModelCheckpoint`` callback, so that the ``Trainer`` recognizes the custom class as a checkpointing callback.


***********************
Customize Checkpointing
***********************

.. warning::

    The Checkpoint IO API is experimental and subject to change.


Lightning supports modifying the checkpointing save/load functionality through the ``CheckpointIO``. This encapsulates the save/load logic
that is managed by the ``Strategy``. ``CheckpointIO`` is different from :meth:`~pytorch_lightning.core.hooks.CheckpointHooks.on_save_checkpoint`
and :meth:`~pytorch_lightning.core.hooks.CheckpointHooks.on_load_checkpoint` methods as it determines how the checkpoint is saved/loaded to storage rather than
what's saved in the checkpoint.


TODO: I don't understand this...

******************************
Built-in Checkpoint IO Plugins
******************************

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
   * - :class:`~pytorch_lightning.plugins.io.HPUCheckpointIO`
     - CheckpointIO to save checkpoints for HPU training strategies.
   * - :class:`~pytorch_lightning.plugins.io.AsyncCheckpointIO`
     - ``AsyncCheckpointIO`` enables saving the checkpoints asynchronously in a thread.


***************************
Custom Checkpoint IO Plugin
***************************

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

    Some ``Strategy``s like ``DeepSpeedStrategy`` do not support custom ``CheckpointIO`` as checkpointing logic is not modifiable.


**************************
Asynchronous Checkpointing
**************************

.. warning::

    This is currently an experimental plugin/feature and API changes are to be expected.

To enable saving the checkpoints asynchronously without blocking your training, you can configure
:class:`~pytorch_lightning.plugins.io.async_plugin.AsyncCheckpointIO` plugin to ``Trainer``.

.. code-block:: python

   from pytorch_lightning.plugins.io import AsyncCheckpointIO


   async_ckpt_io = AsyncCheckpointIO()
   trainer = Trainer(plugins=[async_ckpt_io])


It uses its base ``CheckpointIO`` plugin's saving logic to save the checkpoint but performs this operation asynchronously.
By default, this base ``CheckpointIO`` will be set-up for you and all you need to provide is the ``AsyncCheckpointIO`` instance to the ``Trainer``.
But if you want the plugin to use your own custom base ``CheckpointIO`` and want the base to behave asynchronously, pass it as an argument while initializing ``AsyncCheckpointIO``.

.. code-block:: python

   from pytorch_lightning.plugins.io import AsyncCheckpointIO

   base_ckpt_io = MyCustomCheckpointIO()
   async_ckpt_io = AsyncCheckpointIO(checkpoint_io=base_ckpt_io)
   trainer = Trainer(plugins=[async_ckpt_io])
