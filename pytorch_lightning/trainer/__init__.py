"""
# Trainer

The lightning trainer abstracts best practices for running a training, val, test routine.
 It calls parts of your model when it wants to hand over full control and otherwise makes
 training assumptions which are now standard practice in AI research.

This is the basic use of the trainer:

.. code-block:: python

    from pytorch_lightning import Trainer

    model = LightningTemplate()

    trainer = Trainer()
    trainer.fit(model)

The Trainer holds all the engineering code you might need such as distributing over GPUs or early stopping.
The LightningTemplate holds the core computations, train, val, test loop, optimizer and dataloaders.

This pattern de-couples the engineering from the science which makes your code reusable and free to run on any hardware.
"""