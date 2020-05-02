.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule


Experiment Logging
==================

Comet.ml
^^^^^^^^

`Comet.ml <https://www.comet.ml/site/>`_ is a third-party logger.
To use :class:`~pytorch_lightning.loggers.CometLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install comet-ml

Then configure the logger and pass it to the :class:`~pytorch_lightning.trainer.trainer.Trainer`:

.. testcode::

    import os
    from pytorch_lightning.loggers import CometLogger
    comet_logger = CometLogger(
        api_key=os.environ.get('COMET_API_KEY'),
        workspace=os.environ.get('COMET_WORKSPACE'),  # Optional
        save_dir='.',  # Optional
        project_name='default_project',  # Optional
        rest_api_key=os.environ.get('COMET_REST_API_KEY'),  # Optional
        experiment_name='default'  # Optional
    )
    trainer = Trainer(logger=comet_logger)

The :class:`~pytorch_lightning.loggers.CometLogger` is available anywhere except ``__init__`` in your
:class:`~pytorch_lightning.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            self.logger.experiment.add_image('generated_images', some_img, 0)

.. seealso::
    :class:`~pytorch_lightning.loggers.CometLogger` docs.

MLflow
^^^^^^

`MLflow <https://mlflow.org/>`_ is a third-party logger.
To use :class:`~pytorch_lightning.loggers.MLFlowLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install mlflow

Then configure the logger and pass it to the :class:`~pytorch_lightning.trainer.trainer.Trainer`:

.. testcode::

    from pytorch_lightning.loggers import MLFlowLogger
    mlf_logger = MLFlowLogger(
        experiment_name="default",
        tracking_uri="file:./ml-runs"
    )
    trainer = Trainer(logger=mlf_logger)

.. seealso::
    :class:`~pytorch_lightning.loggers.MLFlowLogger` docs.

Neptune.ai
^^^^^^^^^^

`Neptune.ai <https://neptune.ai/>`_ is a third-party logger.
To use :class:`~pytorch_lightning.loggers.NeptuneLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install neptune-client

Then configure the logger and pass it to the :class:`~pytorch_lightning.trainer.trainer.Trainer`:

.. testcode::

    from pytorch_lightning.loggers import NeptuneLogger
    neptune_logger = NeptuneLogger(
        api_key='ANONYMOUS',  # replace with your own
        project_name='shared/pytorch-lightning-integration',
        experiment_name='default',  # Optional,
        params={'max_epochs': 10},  # Optional,
        tags=['pytorch-lightning', 'mlp'],  # Optional,
    )
    trainer = Trainer(logger=neptune_logger)

The :class:`~pytorch_lightning.loggers.NeptuneLogger` is available anywhere except ``__init__`` in your
:class:`~pytorch_lightning.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            self.logger.experiment.add_image('generated_images', some_img, 0)

.. seealso::
    :class:`~pytorch_lightning.loggers.NeptuneLogger` docs.

allegro.ai TRAINS
^^^^^^^^^^^^^^^^^

`allegro.ai <https://github.com/allegroai/trains/>`_ is a third-party logger.
To use :class:`~pytorch_lightning.loggers.TrainsLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install trains

Then configure the logger and pass it to the :class:`~pytorch_lightning.trainer.trainer.Trainer`:

.. testcode::

    from pytorch_lightning.loggers import TrainsLogger
    trains_logger = TrainsLogger(
        project_name='examples',
        task_name='pytorch lightning test',
    )
    trainer = Trainer(logger=trains_logger)

.. testoutput::
    :options: +ELLIPSIS, +NORMALIZE_WHITESPACE
    :hide:

    TRAINS Task: ...
    TRAINS results page: ...

The :class:`~pytorch_lightning.loggers.TrainsLogger` is available anywhere in your
:class:`~pytorch_lightning.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def __init__(self):
            some_img = fake_image()
            self.logger.experiment.log_image('debug', 'generated_image_0', some_img, 0)

.. seealso::
    :class:`~pytorch_lightning.loggers.TrainsLogger` docs.

Tensorboard
^^^^^^^^^^^

To use `TensorBoard <https://pytorch.org/docs/stable/tensorboard.html>`_ as your logger do the following.

.. testcode::

    from pytorch_lightning.loggers import TensorBoardLogger
    logger = TensorBoardLogger('tb_logs', name='my_model')
    trainer = Trainer(logger=logger)

The :class:`~pytorch_lightning.loggers.TensorBoardLogger` is available anywhere except ``__init__`` in your
:class:`~pytorch_lightning.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            self.logger.experiment.add_image('generated_images', some_img, 0)

.. seealso::
    :class:`~pytorch_lightning.loggers.TensorBoardLogger` docs.

Test Tube
^^^^^^^^^

`Test Tube <https://github.com/williamFalcon/test-tube>`_ is a
`TensorBoard <https://pytorch.org/docs/stable/tensorboard.html>`_  logger but with nicer file structure.
To use :class:`~pytorch_lightning.loggers.TestTubeLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install test_tube

Then configure the logger and pass it to the :class:`~pytorch_lightning.trainer.trainer.Trainer`:

.. testcode::

    from pytorch_lightning.loggers import TestTubeLogger
    logger = TestTubeLogger('tb_logs', name='my_model')
    trainer = Trainer(logger=logger)

The :class:`~pytorch_lightning.loggers.TestTubeLogger` is available anywhere except ``__init__`` in your
:class:`~pytorch_lightning.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            self.logger.experiment.add_image('generated_images', some_img, 0)

.. seealso::
    :class:`~pytorch_lightning.loggers.TestTubeLogger` docs.

Weights and Biases
^^^^^^^^^^^^^^^^^^

`Weights and Biases <https://www.wandb.com/>`_ is a third-party logger.
To use :class:`~pytorch_lightning.loggers.WandbLogger` as your logger do the following.
First, install the package:

.. code-block:: bash

    pip install wandb

Then configure the logger and pass it to the :class:`~pytorch_lightning.trainer.trainer.Trainer`:

.. testcode::

    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger()
    trainer = Trainer(logger=wandb_logger)

The :class:`~pytorch_lightning.loggers.WandbLogger` is available anywhere except ``__init__`` in your
:class:`~pytorch_lightning.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            self.logger.experiment.log({
                 "generated_images": [wandb.Image(some_img, caption="...")]
            })

.. seealso::
    :class:`~pytorch_lightning.loggers.WandbLogger` docs.

Multiple Loggers
^^^^^^^^^^^^^^^^

Lightning supports the use of multiple loggers, just pass a list to the
:class:`~pytorch_lightning.trainer.trainer.Trainer`.

.. testcode::

    from pytorch_lightning.loggers import TensorBoardLogger, TestTubeLogger
    logger1 = TensorBoardLogger('tb_logs', name='my_model')
    logger2 = TestTubeLogger('tb_logs', name='my_model')
    trainer = Trainer(logger=[logger1, logger2])
   
The loggers are available as a list anywhere except ``__init__`` in your
:class:`~pytorch_lightning.core.lightning.LightningModule`.

.. testcode::

    class MyModule(LightningModule):
        def any_lightning_module_function_or_hook(self):
            some_img = fake_image()
            # Option 1
            self.logger.experiment[0].add_image('generated_images', some_img, 0)
            # Option 2
            self.logger[0].experiment.add_image('generated_images', some_img, 0)
