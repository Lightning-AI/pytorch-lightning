"""

The trainer de-couples the engineering code (16-bit, early stopping, GPU distribution, etc...) from the
science code (GAN, BERT, your project, etc...). It uses many assumptions which are best practices in
AI research today.

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
