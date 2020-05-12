.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer


Experiment Reporting
=====================

Lightning supports many different experiment loggers. These loggers allow you to monitor losses, images, text, etc...
as training progresses. They usually provide a GUI to visualize and can sometimes even snapshot hyperparameters
used in each experiment.


Control logging frequency
^^^^^^^^^^^^^^^^^^^^^^^^^

It may slow training down to log every single batch. Trainer has an option to log every k batches instead.

.. testcode::

   k = 10
   trainer = Trainer(row_log_interval=k)

Control log writing frequency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Writing to a logger  can be expensive. In Lightning you can set the interval at which you
want to log using this trainer flag.

.. seealso::
    :class:`~pytorch_lightning.trainer.trainer.Trainer`

.. testcode::

    k = 100
    trainer = Trainer(log_save_interval=k)

Log metrics
^^^^^^^^^^^

To plot metrics into whatever logger you passed in (tensorboard, comet, neptune, TRAINS, etc...)

1. training_epoch_end, validation_epoch_end, test_epoch_end will all log anything in the "log" key of the return dict.

.. testcode::

    def training_epoch_end(self, outputs):
        loss = some_loss()
        ...

        logs = {'train_loss': loss}
        results = {'log': logs}
        return results

    def validation_epoch_end(self, outputs):
        loss = some_loss()
        ...

        logs = {'val_loss': loss}
        results = {'log': logs}
        return results

    def test_epoch_end(self, outputs):
        loss = some_loss()
        ...

        logs = {'test_loss': loss}
        results = {'log': logs}
        return results

2. In addition, you can also use any arbitrary functionality from a particular logger from within your LightningModule.
For instance, here we log images using tensorboard.

.. testcode::
    :skipif: not TORCHVISION_AVAILABLE

    def training_step(self, batch, batch_idx):
        self.generated_imgs = self.decoder.generate()

        sample_imgs = self.generated_imgs[:6]
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image('generated_images', grid, 0)

        ...
        return results

Modify progress bar
^^^^^^^^^^^^^^^^^^^

Each return dict from the training_end, validation_end, testing_end and training_step also has
a key called "progress_bar".

Here we show the validation loss in the progress bar

.. testcode::

    def validation_epoch_end(self, outputs):
        loss = some_loss()
        ...

        logs = {'val_loss': loss}
        results = {'progress_bar': logs}
        return results

Snapshot hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^
When training a model, it's useful to know what hyperparams went into that model.
When Lightning creates a checkpoint, it stores a key "hparams" with the hyperparams.

.. code-block:: python

    lightning_checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    hyperparams = lightning_checkpoint['hparams']

Some loggers also allow logging the hyperparams used in the experiment. For instance,
when using the TestTubeLogger or the TensorBoardLogger, all hyperparams will show
in the `hparams tab <https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams>`_.

Snapshot code
^^^^^^^^^^^^^
Loggers  also allow you to snapshot a copy of the code used in this experiment.
For example, TestTubeLogger does this with a flag:

.. testcode::

    from pytorch_lightning.loggers import TestTubeLogger
    logger = TestTubeLogger('.', create_git_tag=True)
