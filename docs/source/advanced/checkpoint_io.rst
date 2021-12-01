Checkpointing IO
================

.. warning:: The Checkpoint IO API is experimental and subject to change.

Lightning supports modifying the checkpointing save/load functionality through the ``CheckpointIO``. This encapsulates the save/load logic
that is managed by the ``TrainingTypePlugin``. ``CheckpointIO`` is different from :meth:`pytorch_lightning.core.lightning.LightningModule.on_save_checkpoint` method
as it determines how the checkpoint is saved/loaded rather than what's saved in the checkpoint.

-----------

Built-in Checkpoint IO Plugins
-------------------------------

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

-----------

Custom Checkpoint IO Plugin
---------------------------
``CheckpointIO`` can be extended to include your custom save/load functionality to and from a path. The ``CheckpointIO`` object can be passed to either a ``Trainer`` directly or a ``TrainingTypePlugin`` as shown below:

.. code-block:: python

    from pathlib import Path
    from typing import Any, Dict, Optional, Union

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.plugins import CheckpointIO, SingleDevicePlugin


    class CustomCheckpointIO(CheckpointIO):
        def save_checkpoint(
            self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
        ) -> None:
            ...

        def load_checkpoint(self, path: Union[str, Path], storage_options: Optional[Any] = None) -> Dict[str, Any]:
            ...

        def remove_checkpoint(self, path: _PATH) -> None:
            ...


    custom_checkpoint_io = CustomCheckpointIO()

    # Either pass into the Trainer object
    model = MyModel()
    trainer = Trainer(
        plugins=[custom_checkpoint_io],
        callbacks=ModelCheckpoint(save_last=True),
    )
    trainer.fit(model)

    # or pass into TrainingTypePlugin
    model = MyModel()
    device = torch.device("cpu")
    trainer = Trainer(
        plugins=SingleDevicePlugin(device, checkpoint_io=custom_checkpoint_io),
        callbacks=ModelCheckpoint(save_last=True),
    )
    trainer.fit(model)

.. note::

    Some ``TrainingTypePlugins`` for eg. ``DeepSpeedPlugin`` do not support custom ``CheckpointIO`` as checkpointing logic is not modifiable.
