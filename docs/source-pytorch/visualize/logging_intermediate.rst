.. _logging_intermediate:

##############################################
Track and Visualize Experiments (intermediate)
##############################################
**Audience:** Users who want to track more complex outputs and use third-party experiment managers.

----

*******************************
Track audio and other artifacts
*******************************
To track other artifacts, such as histograms or model topology graphs first select one of the many loggers supported by Lightning

.. code-block:: python

    from lightning.pytorch import loggers as pl_loggers

    tensorboard = pl_loggers.TensorBoardLogger(save_dir="")
    trainer = Trainer(logger=tensorboard)

then access the logger's API directly

.. code-block:: python

    def training_step(self):
        tensorboard = self.logger.experiment
        tensorboard.add_image()
        tensorboard.add_histogram(...)
        tensorboard.add_figure(...)

----

.. include:: supported_exp_managers.rst

----

*********************
Track hyperparameters
*********************
To track hyperparameters, first call *save_hyperparameters* from the LightningModule init:

.. code-block:: python

    class MyLightningModule(LightningModule):
        def __init__(self, learning_rate, another_parameter, *args, **kwargs):
            super().__init__()
            self.save_hyperparameters()

If your logger supports tracked hyperparameters, the hyperparameters will automatically show up on the logger dashboard.

.. TODO:: show tracked hyperparameters.

----

********************
Track model topology
********************
Multiple loggers support visualizing the model topology. Here's an example that tracks the model topology using Tensorboard.

.. code-block:: python

    def any_lightning_module_function_or_hook(self):
        tensorboard_logger = self.logger

        prototype_array = torch.Tensor(32, 1, 28, 27)
        tensorboard_logger.log_graph(model=self, input_array=prototype_array)

.. TODO:: show tensorboard topology.
