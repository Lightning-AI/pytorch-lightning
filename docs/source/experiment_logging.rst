Experiment Logging
===================

Comet.ml
^^^^^^^^

`Comet.ml <https://www.comet.ml/site/>`_ is a third-party logger.
To use CometLogger as your logger do the following.

.. seealso::
    :class:`~pytorch_lightning.loggers.CometLogger` docs.

.. code-block:: python

   from pytorch_lightning.loggers import CometLogger

    comet_logger = CometLogger(
        api_key=os.environ["COMET_KEY"],
        workspace=os.environ["COMET_WORKSPACE"], # Optional
        project_name="default_project", # Optional
        rest_api_key=os.environ["COMET_REST_KEY"], # Optional
        experiment_name="default" # Optional
    )
   trainer = Trainer(logger=comet_logger)

The CometLogger is available anywhere except ``__init__`` in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def any_lightning_module_function_or_hook(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

Neptune.ai
^^^^^^^^^^

`Neptune.ai <https://neptune.ai/>`_ is a third-party logger.
To use Neptune.ai as your logger do the following.

.. seealso::
    :class:`~pytorch_lightning.loggers.NeptuneLogger` docs.

.. code-block:: python

   from pytorch_lightning.loggers import NeptuneLogger

    neptune_logger = NeptuneLogger(
        project_name="USER_NAME/PROJECT_NAME",
        experiment_name="default", # Optional,
        params={"max_epochs": 10}, # Optional,
        tags=["pytorch-lightning","mlp"] # Optional,
    )
   trainer = Trainer(logger=neptune_logger)

The Neptune.ai is available anywhere except ``__init__`` in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def any_lightning_module_function_or_hook(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

allegro.ai TRAINS
^^^^^^^^^^^^^^^^^

`allegro.ai <https://github.com/allegroai/trains/>`_ is a third-party logger.
To use TRAINS as your logger do the following.

.. seealso::
    :class:`~pytorch_lightning.loggers.TrainsLogger` docs.

.. code-block:: python

   from pytorch_lightning.loggers import TrainsLogger

    trains_logger = TrainsLogger(
        project_name="examples",
        task_name="pytorch lightning test"
    )
   trainer = Trainer(logger=trains_logger)

The TrainsLogger is available anywhere in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def __init__(self, ...):
         some_img = fake_image()
         self.logger.log_image('debug', 'generated_image_0', some_img, 0)

Tensorboard
^^^^^^^^^^^

To use `Tensorboard <https://pytorch.org/docs/stable/tensorboard.html>`_ as your logger do the following.

.. seealso::
    :class:`~pytorch_lightning.loggers.TensorBoardLogger` docs.

.. code-block:: python

   from pytorch_lightning.loggers import TensorBoardLogger

   logger = TensorBoardLogger("tb_logs", name="my_model")
   trainer = Trainer(logger=logger)

The TensorBoardLogger is available anywhere except ``__init__`` in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def any_lightning_module_function_or_hook(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)


Test Tube
^^^^^^^^^

`Test Tube <https://github.com/williamFalcon/test-tube>`_ is a tensorboard logger but with nicer file structure.
To use TestTube as your logger do the following.

.. seealso::
    :class:`~pytorch_lightning.loggers.TestTubeLogger` docs.

.. code-block:: python

   from pytorch_lightning.loggers import TestTubeLogger

   logger = TestTubeLogger("tb_logs", name="my_model")
   trainer = Trainer(logger=logger)

The TestTubeLogger is available anywhere except ``__init__`` in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def any_lightning_module_function_or_hook(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)

Wandb
^^^^^

`Wandb <https://www.wandb.com/>`_ is a third-party logger.
To use Wandb as your logger do the following.

.. seealso::
    :class:`~pytorch_lightning.loggers.WandbLogger` docs.

.. code-block:: python

   from pytorch_lightning.loggers import WandbLogger

   wandb_logger = WandbLogger()
   trainer = Trainer(logger=wandb_logger)

The Wandb logger is available anywhere except ``__init__`` in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def any_lightning_module_function_or_hook(self, ...):
         some_img = fake_image()
         self.logger.experiment.add_image('generated_images', some_img, 0)


Multiple Loggers
^^^^^^^^^^^^^^^^

PyTorch-Lightning supports use of multiple loggers, just pass a list to the `Trainer`.

.. code-block:: python

   from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger
   
   logger1 = TensorBoardLogger("tb_logs", name="my_model")
   logger2 = TestTubeLogger("tt_logs", name="my_model")
   trainer = Trainer(logger=[logger1, logger2])
   
The loggers are available as a list anywhere except ``__init__`` in your LightningModule

.. code-block:: python

   class MyModule(pl.LightningModule):

      def any_lightning_module_function_or_hook(self, ...):
         some_img = fake_image()

         # Option 1
         self.logger.experiment[0].add_image('generated_images', some_img, 0)

         # Option 2
         self.logger[0].experiment.add_image('generated_images', some_img, 0)
