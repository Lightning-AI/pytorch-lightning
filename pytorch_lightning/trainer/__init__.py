"""
Once you've organized your PyTorch code into a LightningModule,
the Trainer automates everything else.

.. figure:: /_images/lightning_module/pt_to_trainer.png
   :alt: Convert from PyTorch to Lightning


The trainer automates all parts of training except:

- what happens in training , test, val loop
- where the data come from
- which optimizers to use
- how to do the computations

The Trainer delegates those calls to your LightningModule which defines how to do those parts.

This is the basic use of the trainer:

.. code-block:: python

    from pytorch_lightning import Trainer

    model = MyLightningModule()

    trainer = Trainer()
    trainer.fit(model)
"""

from .trainer import Trainer

__all__ = ['Trainer']
