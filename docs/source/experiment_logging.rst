Experiment Logging and Reporting
=================================

Lightning supports many different experiment loggers. These loggers allow you to monitor losses, images, text, etc...
as training progresses. They usually provide a GUI to visualize and can sometimes even snapshot hyperparameters
used in each experiment.

Control logging frequency
----------------------------------------------
It may slow training down to log every single batch. Trainer has an option to log every k batches instead.

.. code-block:: python

   # k = 10
   Trainer(row_log_interval=10)

Control log writing frequency
----------------------------------------------
Writing to a logger  can be expensive. In Lightning you can set the interval at which you
want to log using this trainer flag.

.. note:: See: :ref:`trainer`

.. code-block:: python

   k = 100
   Trainer(log_save_interval=k)

Log metrics
----------------------------------------------

To plot metrics into whatever logger you passed in (tensorboard, comet, neptune, etc...)

1. Training_end, validation_end, test_end will all log anything in the "log" key of the return dict.

.. code-block:: python

   def training_end(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'train_loss': loss}
      results = {'log': logs}
      return results

   def validation_end(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'val_loss': loss}
      results = {'log': logs}
      return results

   def test_end(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'test_loss': loss}
      results = {'log': logs}
      return results

2. Most of the time, you only need training_step and not training_end. You can also return logs from here:

.. code-block:: python

   def training_step(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'train_loss': loss}
      results = {'log': logs}
      return results

3. In addition, you can also use any arbitrary functionality from a particular logger from within your LightningModule.
For instance, here we log images using tensorboard.

.. code-block:: python

   def training_step(self, batch, batch_idx):
      self.generated_imgs = self.decoder.generate()

      sample_imgs = self.generated_imgs[:6]
      grid = torchvision.utils.make_grid(sample_imgs)
      self.logger.experiment.add_image('generated_images', grid, 0)

      ...
      return results

Modify progress bar
----------------------------
Each return dict from the training_end, validation_end, testing_end and training_step also has
a key called "progress_bar".

Here we show the validation loss in the progress bar

.. code-block:: python

   def validation_end(self, batch, batch_idx):
      loss = some_loss()
      ...

      logs = {'val_loss': loss}
      results = {'progress_bar': logs}
      return results

Snapshot hyperparameters
----------------------------------------------
When training a model, it's useful to know what hyperparams went into that model.
When Lightning creates a checkpoint, it stores a key "hparams" with the hyperparams.

.. code-block:: python

   lightning_checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
   hyperparams = lightning_checkpoint['hparams']

Some loggers also allow logging the hyperparams used in the experiment. For instance,
when using the TestTubeLogger or the TensorBoardLogger, all hyperparams will show
in the `hparams tab <https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_hparams>`_.

Snapshot code
----------------------------------------------
Loggers  also allow you to snapshot a copy of the code used in this experiment.
For example, TestTubeLogger does this with a flag:

.. code-block:: python

   from pytorch_lightning.loggers import TestTubeLogger

   logger = TestTubeLogger(create_git_tag=True)

Experiment Loggers
=================================

Comet.ml
----------------------------------------------
`Comet.ml <https://www.comet.ml/site/>`_ is a third-party logger.
To use CometLogger as your logger do the following.

.. note:: See: :ref:`comet` docs.

.. code-block:: python

   from pytorch_lightning.loggers import TestTubeLogger

    comet_logger = CometLogger(
        api_key=os.environ["COMET_KEY"],
        workspace=os.environ["COMET_WORKSPACE"], # Optional
        project_name="default_project", # Optional
        rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        experiment_name="default" # Optional
    )
   trainer = Trainer(logger=comet_logger)

The CometLogger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

Neptune.ai
----------------------------------------------
`Neptune.ai <https://neptune.ai/>`_ is a third-party logger.
To use Neptune.ai as your logger do the following.

.. note:: See: :ref:`neptune` docs.

.. code-block:: python

   from pytorch_lightning.loggers import NeptuneLogger

    neptune_logger = NeptuneLogger(
        project_name="USER_NAME/PROJECT_NAME",
        experiment_name="default", # Optional,
        params={"max_epochs": 10}, # Optional,
        tags=["pytorch-lightning","mlp"] # Optional,
    )
   trainer = Trainer(logger=neptune_logger)

The Neptune.ai is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

Tensorboard
----------------------------------------------
To use `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ as your logger do the following.

.. note:: See: TensorBoardLogger :ref:`tf-logger`

.. code-block:: python

   from pytorch_lightning.loggers import TensorBoardLogger

   logger = TensorBoardLogger("tb_logs", name="my_model")
   trainer = Trainer(logger=logger)

The TensorBoardLogger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)


Test Tube
----------------------------------------------
`Test Tube <https://github.com/williamFalcon/test-tube>`_ is a tensorboard logger but with nicer file structure.
To use TestTube as your logger do the following.

.. note:: See: TestTube :ref:`testTube`

.. code-block:: python

   from pytorch_lightning.loggers import TestTubeLogger

   logger = TestTubeLogger("tb_logs", name="my_model")
   trainer = Trainer(logger=logger)

The TestTubeLogger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

Wandb
----------------------------------------------
`Wandb <https://www.wandb.com/>`_ is a third-party logger.
To use Wandb as your logger do the following.

.. note:: See: :ref:`wandb` docs

.. code-block:: python

   from pytorch_lightning.loggers import WandbLogger

   wandb_logger = WandbLogger()
   trainer = Trainer(logger=wandb_logger)

The Wandb logger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

