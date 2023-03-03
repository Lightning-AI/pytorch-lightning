******************
Manage Experiments
******************
To track other artifacts, such as histograms or model topology graphs first select one of the many experiment managers (*loggers*) supported by Lightning

.. code-block:: python

    from lightning.pytorch import loggers as pl_loggers

    tensorboard = pl_loggers.TensorBoardLogger()
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
