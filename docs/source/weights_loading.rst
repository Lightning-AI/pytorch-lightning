Saving and loading weights
==========================

Lightning can automate saving and loading checkpoints.

Checkpoint saving
-----------------
A Lightning checkpoint has everything needed to restore a training session including:

- 16-bit scaling factor (apex)
- Current epoch
- Global step
- Model state_dict
- State of all optimizers
- State of all learningRate schedulers
- State of all callbacks
- The hyperparameters used for that model if passed in as hparams (Argparse.Namespace)

Automatic saving
^^^^^^^^^^^^^^^^

Checkpointing is enabled by default to the current working directory.
To change the checkpoint path pass in:

.. code-block:: python

    Trainer(default_save_path='/your/path/to/save/checkpoints')

To modify the behavior of checkpointing pass in your own callback.

.. code-block:: python

    from pytorch_lightning.callbacks import ModelCheckpoint

    # DEFAULTS used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=os.getcwd(),
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    trainer = Trainer(checkpoint_callback=checkpoint_callback)


Or disable it by passing

.. code-block:: python

        trainer = Trainer(checkpoint_callback=False)


The Lightning checkpoint also saves the hparams (hyperparams) passed into the LightningModule init.

.. note:: hparams is a `Namespace <https://docs.python.org/2/library/argparse.html#argparse.Namespace>`_.

.. code-block:: python
   :emphasize-lines: 8

   from argparse import Namespace

   # usually these come from command line args
   args = Namespace(learning_rate=0.001)

   # define you module to have hparams as the first arg
   # this means your checkpoint will have everything that went into making
   # this model (in this case, learning rate)
   class MyLightningModule(pl.LightningModule):

       def __init__(self, hparams, ...):
           self.hparams = hparams

Manual saving
^^^^^^^^^^^^^

To save your own checkpoint call:

.. code-block:: python

   model.save_checkpoint(PATH)

Checkpoint Loading
------------------

To load a model along with its weights, biases and hyperparameters use following method.

.. code-block:: python

    model = MyLightingModule.load_from_checkpoint(PATH)
    model.eval()
    y_hat = model(x)

The above only works if you used `hparams` in your model definition

.. code-block:: python

    class MyModel(pl.LightningModule):

        def __init__(self, hparams):
            self.hparams = hparams
            self.l1 = nn.Linear(hparams.in_dim, hparams.out_dim)

But if you don't and instead pass individual parameters

.. code-block:: python

    class MyModel(pl.LightningModule):

        def __init__(self, in_dim, out_dim):
            self.l1 = nn.Linear(in_dim, out_dim)

you can restore the model like this

.. code-block:: python

    model = MyModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)


Restoring Training State
------------------------

If you don't just want to load weights, but instead restore the full training,
do the following:

.. code-block:: python

   model = LitModel()
   trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

   # automatically restores model, epoch, step, LR schedulers, apex, etc...
   trainer.fit(model)
