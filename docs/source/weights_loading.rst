.. testsetup:: *

    import os
    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _weights_loading:

##########################
Saving and loading weights
##########################

Lightning automates saving and loading checkpoints. Checkpoints capture the exact value of all parameters used by a model.

Checkpointing your training allows you to resume a training process in case it was interrupted, fine-tune a model or use a pre-trained model for inference without having to retrain the model.

*****************
Checkpoint saving
*****************
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
================

Lightning automatically saves a checkpoint for you in your current working directory, with the state of your last training epoch. This makes sure you can resume training in case it was interrupted.

To change the checkpoint path pass in:

.. code-block:: python

    # saves checkpoints to '/your/path/to/save/checkpoints' at every epoch end
    trainer = Trainer(default_root_dir='/your/path/to/save/checkpoints')

You can customize the checkpointing behavior to monitor any quantity of your training or validation steps. For example, if you want to update your checkpoints based on your validation loss:

1. Calculate any metric or other quantity you wish to monitor, such as validation loss.
2. Log the quantity using :func:`~~pytorch_lightning.core.lightning.LightningModule.log` method, with a key such as `val_loss`.
3. Initializing the :class:`~pytorch_lightning.callbacks.ModelCheckpoint` callback, and set `monitor` to be the key of your quantity.
4. Pass the callback to `checkpoint_callback` :class:`~pytorch_lightning.trainer.Trainer` flag.

.. code-block:: python

    from pytorch_lightning.callbacks import ModelCheckpoint

    class LitAutoEncoder(pl.LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.backbone(x)

            # 1. calculate loss
            loss = F.cross_entropy(y_hat, y)

            # 2. log `val_loss`
            self.log('val_loss', loss)

    # 3. Init ModelCheckpoint callback, monitoring 'val_loss'
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')

    # 4. Pass your callback to checkpoint_callback trainer flag
    trainer = Trainer(checkpoint_callback=checkpoint_callback)

You can also control more advanced options, like `save_top_k`, to save the best k models and the mode of the monitored quantity (min/max/auto, where the mode is automatically inferred from the name of the monitored quantity), `save_weights_only` or `period` to set the interval of epochs between checkpoints, to avoid slowdowns.

.. code-block:: python

    from pytorch_lightning.callbacks import ModelCheckpoint

    class LitAutoEncoder(pl.LightningModule):
        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.backbone(x)
            loss = F.cross_entropy(y_hat, y)
            self.log('val_loss', loss)

    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='my/path/,
        filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min')

    trainer = Trainer(checkpoint_callback=checkpoint_callback)
    
You can retrieve the checkpoint after training by calling

.. code-block:: python

        checkpoint_callback = ModelCheckpoint(dirpath='my/path/')
        trainer = Trainer(checkpoint_callback=checkpoint_callback)
        trainer.fit(model)
        checkpoint_callback.best_model_path

Disabling checkpoints
---------------------

You can disable checkpointing by passing

.. testcode::

   trainer = Trainer(checkpoint_callback=False)


The Lightning checkpoint also saves the arguments passed into the LightningModule init
under the `module_arguments` key in the checkpoint.

.. code-block:: python

    class MyLightningModule(LightningModule):

       def __init__(self, learning_rate, *args, **kwargs):
            super().__init__()

    # all init args were saved to the checkpoint
    checkpoint = torch.load(CKPT_PATH)
    print(checkpoint['module_arguments'])
    # {'learning_rate': the_value}

Manual saving
=============
You can manually save checkpoints and restore your model from the checkpointed state.

.. code-block:: python

    model = MyLightningModule(hparams)
    trainer.fit(model)
    trainer.save_checkpoint("example.ckpt")
    new_model = MyModel.load_from_checkpoint(checkpoint_path="example.ckpt")

******************
Checkpoint loading
******************

To load a model along with its weights, biases and `module_arguments` use the following method:

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
            self.save_hyperparameters()
            self.l1 = nn.Linear(self.hparams.in_dim, self.hparams.out_dim)

you can restore the model like this

.. code-block:: python

    # if you train and save the model like this it will use these values when loading
    # the weights. But you can overwrite this
    LitModel(in_dim=32, out_dim=10)

    # uses in_dim=32, out_dim=10
    model = LitModel.load_from_checkpoint(PATH)

    # uses in_dim=128, out_dim=10
    model = LitModel.load_from_checkpoint(PATH, in_dim=128, out_dim=10)

.. automethod:: pytorch_lightning.core.lightning.LightningModule.load_from_checkpoint
   :noindex:

Restoring Training State
========================

If you don't just want to load weights, but instead restore the full training,
do the following:

.. code-block:: python

   model = LitModel()
   trainer = Trainer(resume_from_checkpoint='some/path/to/my_checkpoint.ckpt')

   # automatically restores model, epoch, step, LR schedulers, apex, etc...
   trainer.fit(model)
