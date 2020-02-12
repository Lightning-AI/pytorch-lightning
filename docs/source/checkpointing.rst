Checkpointing
==============

.. _model-saving:

Model saving
-------------------
To save a LightningModule, provide a :meth:`pytorch_lightning.callbacks.ModelCheckpoint` callback.

The Lightning checkpoint also saves the hparams (hyperparams) passed into the LightningModule init.

.. note:: hparams is a `Namespace <https://docs.python.org/2/library/argparse.html#argparse.Namespace>`_ or dictionary.

.. code-block:: python
   :emphasize-lines: 8

   from argparse import Namespace

   # usually these come from command line args
   args = Namespace(**{'learning_rate':0.001})

   # define you module to have hparams as the first arg
   # this means your checkpoint will have everything that went into making
   # this model (in this case, learning rate)
   class MyLightningModule(pl.LightningModule):

       def __init__(self, hparams, ...):
           self.hparams = hparams

   my_model = MyLightningModule(args)

   # auto-saves checkpoint
   checkpoint_callback = ModelCheckpoint(filepath='my_path')
   Trainer(checkpoint_callback=checkpoint_callback)


Model loading
-----------------------------------

To load a model, use :meth:`pytorch_lightning.core.LightningModule.load_from_checkpoint`

.. note:: If lightning created your checkpoint, your model will receive all the hyperparameters used
   to create the checkpoint. (See: :ref:`model-saving`).

.. code-block:: python

    # load weights without mapping
    MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

    # load weights mapping all weights from GPU 1 to GPU 0
    map_location = {'cuda:1':'cuda:0'}
    MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt', map_location=map_location)

Restoring training session
-----------------------------------

If you want to pick up training from where you left off, you have a few options.

1. Pass in a logger with the same experiment version to continue training.

.. code-block:: python

   # train the first time and set the version number
   logger = TensorboardLogger(version=10)
   trainer = Trainer(logger=logger)
   trainer.fit(model)

   # when you init another logger with that same version, the model
   # will continue where it left off
   logger = TensorboardLogger(version=10)
   trainer = Trainer(logger=logger)
   trainer.fit(model)

2. A second option is to pass in a path to a checkpoint (see: :ref:`pytorch_lightning.trainer`).

.. code-block:: python

   # train the first time and set the version number
   trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')
   trainer.fit(model)