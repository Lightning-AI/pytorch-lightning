"""
Once you've organized your PyTorch code into a LightningModule,
the Trainer automates everything else.

.. figure:: /_images/lightning_module/pt_trainer.png
   :alt: Convert from PyTorch to Lightning

This abstraction achieves the folowing:

1. You maintain control over all aspects via PyTorch
code without an added abstraction.

2. The trainer uses best practices embedded by contributors and users
from top AI labs such as Facebook AI Research, NYU, MIT, Stanford, etc...

3. The trainer allows overriding any key part that you don't want automated.

-----------

Basic use
---------

This is the basic use of the trainer:

.. code-block:: python

    from pytorch_lightning import Trainer

    model = MyLightningModule()

    trainer = Trainer()
    trainer.fit(model)
"""

from .trainer import Trainer

__all__ = ['Trainer']
