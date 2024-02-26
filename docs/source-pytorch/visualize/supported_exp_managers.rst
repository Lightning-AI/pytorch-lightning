Comet.ml
========
To use `Comet.ml <https://www.comet.ml/site/>`_ first install the comet package:

.. code-block:: bash

    pip install comet-ml

Configure the logger and pass it to the :class:`~lightning.pytorch.trainer.trainer.Trainer`:

.. code-block:: python

    from lightning.pytorch.loggers import CometLogger

    comet_logger = CometLogger(api_key="YOUR_COMET_API_KEY")
    trainer = Trainer(logger=comet_logger)

Access the comet logger from any function (except the LightningModule *init*) to use its API for tracking advanced artifacts

.. code-block:: python

    class LitModel(LightningModule):
        def any_lightning_module_function_or_hook(self):
            comet = self.logger.experiment
            fake_images = torch.Tensor(32, 3, 28, 28)
            comet.add_image("generated_images", fake_images, 0)

Here's the full documentation for the :class:`~lightning.pytorch.loggers.CometLogger`.

----

MLflow
======
To use `MLflow <https://mlflow.org/>`_ first install the MLflow package:

.. code-block:: bash

    pip install mlflow

Configure the logger and pass it to the :class:`~lightning.pytorch.trainer.trainer.Trainer`:

.. code-block:: python

    from lightning.pytorch.loggers import MLFlowLogger

    mlf_logger = MLFlowLogger(experiment_name="lightning_logs", tracking_uri="file:./ml-runs")
    trainer = Trainer(logger=mlf_logger)

Access the mlflow logger from any function (except the LightningModule *init*) to use its API for tracking advanced artifacts

.. code-block:: python

    class LitModel(LightningModule):
        def any_lightning_module_function_or_hook(self):
            mlf_logger = self.logger.experiment
            fake_images = torch.Tensor(32, 3, 28, 28)
            mlf_logger.add_image("generated_images", fake_images, 0)

Here's the full documentation for the :class:`~lightning.pytorch.loggers.MLFlowLogger`.

----

Neptune.ai
==========
To use `Neptune.ai <https://www.neptune.ai/>`_ first install the neptune package:

.. code-block:: bash

    pip install neptune

or with conda:

.. code-block:: bash

    conda install -c conda-forge neptune

Configure the logger and pass it to the :class:`~lightning.pytorch.trainer.trainer.Trainer`:

.. testcode::
    :skipif: not _NEPTUNE_AVAILABLE

    import neptune
    from lightning.pytorch.loggers import NeptuneLogger

    neptune_logger = NeptuneLogger(
        api_key=neptune.ANONYMOUS_API_TOKEN,  # replace with your own
        project="common/pytorch-lightning-integration",  # format "<WORKSPACE/PROJECT>"
    )
    trainer = Trainer(logger=neptune_logger)

Access the neptune logger from any function (except the LightningModule *init*) to use its API for tracking advanced artifacts

.. code-block:: python

    class LitModel(LightningModule):
        def any_lightning_module_function_or_hook(self):
            neptune_logger = self.logger.experiment["your/metadata/structure"]
            neptune_logger.append(metadata)

Here's the full documentation for the :class:`~lightning.pytorch.loggers.NeptuneLogger`.

----

Tensorboard
===========
`TensorBoard <https://pytorch.org/docs/stable/tensorboard.html>`_ can be installed with:

.. code-block:: bash

    pip install tensorboard

Configure the logger and pass it to the :class:`~lightning.pytorch.trainer.trainer.Trainer`:

.. code-block:: python

    from lightning.pytorch.loggers import TensorBoardLogger

    logger = TensorBoardLogger()
    trainer = Trainer(logger=logger)

Access the tensorboard logger from any function (except the LightningModule *init*) to use its API for tracking advanced artifacts

.. code-block:: python

    class LitModel(LightningModule):
        def any_lightning_module_function_or_hook(self):
            tensorboard_logger = self.logger.experiment
            fake_images = torch.Tensor(32, 3, 28, 28)
            tensorboard_logger.add_image("generated_images", fake_images, 0)

Here's the full documentation for the :class:`~lightning.pytorch.loggers.TensorBoardLogger`.

----

Weights and Biases
==================
To use `Weights and Biases <https://docs.wandb.ai/integrations/lightning/>`_ (wandb) first install the wandb package:

.. code-block:: bash

    pip install wandb

Configure the logger and pass it to the :class:`~lightning.pytorch.trainer.trainer.Trainer`:

.. testcode::
    :skipif: not _WANDB_AVAILABLE

    from lightning.pytorch.loggers import WandbLogger

    wandb_logger = WandbLogger(project="MNIST", log_model="all")
    trainer = Trainer(logger=wandb_logger)

    # log gradients and model topology
    wandb_logger.watch(model)

Access the wandb logger from any function (except the LightningModule *init*) to use its API for tracking advanced artifacts

.. code-block:: python

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            wandb_logger = self.logger.experiment
            fake_images = torch.Tensor(32, 3, 28, 28)

            # Option 1
            wandb_logger.log({"generated_images": [wandb.Image(fake_images, caption="...")]})

            # Option 2 for specifically logging images
            wandb_logger.log_image(key="generated_images", images=[fake_images])

Here's the full documentation for the :class:`~lightning.pytorch.loggers.WandbLogger`.
`Demo in Google Colab <http://wandb.me/lightning>`__ with hyperparameter search and model logging.

----

Use multiple exp managers
=========================
To use multiple experiment managers at the same time, pass a list to the *logger* :class:`~lightning.pytorch.trainer.trainer.Trainer` argument.

.. testcode::
    :skipif: (not _TENSORBOARD_AVAILABLE and not _TENSORBOARDX_AVAILABLE) or not _WANDB_AVAILABLE

    from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger

    logger1 = TensorBoardLogger()
    logger2 = WandbLogger()
    trainer = Trainer(logger=[logger1, logger2])


Access all loggers from any function (except the LightningModule *init*) to use their APIs for tracking advanced artifacts

.. code-block:: python

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            tensorboard_logger = self.loggers.experiment[0]
            wandb_logger = self.loggers.experiment[1]

            fake_images = torch.Tensor(32, 3, 28, 28)

            tensorboard_logger.add_image("generated_images", fake_images, 0)
            wandb_logger.add_image("generated_images", fake_images, 0)
