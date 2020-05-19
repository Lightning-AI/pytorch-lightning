.. testsetup:: *

    import os
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule


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

.. testcode::

    trainer = Trainer(default_save_path='/your/path/to/save/checkpoints')

To modify the behavior of checkpointing pass in your own callback.

.. testcode::

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

.. testcode::

   trainer = Trainer(checkpoint_callback=False)


The Lightning checkpoint also saves the arguments passed into the LightningModule init
under the `module_arguments` key in the checkpoint.

.. testcode::

   class MyLightningModule(LightningModule):

       def __init__(self, learning_rate, *args, **kwargs):
            super().__init__()

    # all init args were saved to the checkpoint
    checkpoint = torch.load(CKPT_PATH)
    print(checkpoint['module_arguments'])
    # {'learning_rate': the_value}

Manual saving
^^^^^^^^^^^^^
You can manually save checkpoints and restore your model from the checkpointed state.

.. code-block:: python

    model = MyLightningModule(hparams)
    trainer.fit(model)
    trainer.save_checkpoint("example.ckpt")
    new_model = MyModel.load_from_checkpoint(checkpoint_path="example.ckpt")

Checkpoint Loading
------------------

To load a model along with its weights, biases and model_arguments use following method.

.. code-block:: python

    model = MyLightingModule.load_from_checkpoint(PATH)

    print(model.learning_rate)
    # prints the learning_rate you used in this checkpoint

    model.eval()
    y_hat = model(x)

But if you don't want to use the values saved in the checkpoint, pass in your own here

.. testcode::

    class LitModel(LightningModule):

        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.l1 = nn.Linear(self.in_dim, self.out_dim)

you can restore the model like this

.. code-block:: python

    # if you train and save the model like this it will use these values when loading
    # the weights. But you can overwrite this
    LitModel(in_dim=32, out_dim=10)

    # uses in_dim=32, out_dim=10
    model = LitModel.load_from_checkpoint(PATH)

    # uses in_dim=128, out_dim=10
    model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)


Restoring Training State
------------------------

If you don't just want to load weights, but instead restore the full training,
do the following:

.. code-block:: python

   model = LitModel()
   trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

   # automatically restores model, epoch, step, LR schedulers, apex, etc...
   trainer.fit(model)
