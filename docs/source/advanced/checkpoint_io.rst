Custom Checkpointing IO
=======================

.. warning:: The Checkpoint IO API is experimental and subject to change.

Lightning supports modifying the checkpointing save/load functionality through the ``CheckpointIO``. This encapsulates the save/load logic
that is managed by the ``TrainingTypePlugin``.

``CheckpointIO`` can be extended to include your custom save/load functionality to and from a path. The ``CheckpointIO`` object can be passed to either a `Trainer`` object or a``TrainingTypePlugin`` as shown below.

.. code-block:: python

    from pathlib import Path
    from typing import Any, Dict, Optional, Union

    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.plugins import CheckpointIO, SingleDevicePlugin


    class CustomCheckpointPlugin(CheckpointIO):
        def save_checkpoint(
            self, checkpoint: Dict[str, Any], path: Union[str, Path], storage_options: Optional[Any] = None
        ) -> None:
            ...

        def load_checkpoint(self, path: Union[str, Path], storage_options: Optional[Any] = None) -> Dict[str, Any]:
            ...


    checkpoint_plugin = CustomCheckpointPlugin()

    # Pass into the Trainer object
    model = MyModel()
    trainer = Trainer(
        plugins=[checkpoint_plugin],
        callbacks=ModelCheckpoint(save_last=True),
    )
    trainer.fit(model)

    # pass into TrainingTypePlugin
    model = MyModel()
    device = torch.device("cpu")
    trainer = Trainer(
        plugins=SingleDevicePlugin(device, checkpoint_io=checkpoint_plugin),
        callbacks=ModelCheckpoint(save_last=True),
    )
    trainer.fit(model)

.. note::

    Some ``TrainingTypePlugins`` do not support custom ``CheckpointIO`` as as checkpointing logic is not modifiable.
