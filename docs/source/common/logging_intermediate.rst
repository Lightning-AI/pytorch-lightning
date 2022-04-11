.. testsetup:: *

    from pytorch_lightning.trainer.trainer import Trainer
    from pytorch_lightning.core.lightning import LightningModule

.. _loggers:

##############################
Logging metrics (intermediate)
##############################


*******************************
Track audio and other artifacts
*******************************
To track other artifacts, such as histograms or model topology graphs first select one of the many loggers supported by Lightning 

.. code-block:: python
    from pytorch_lightning import loggers as pl_loggers

    tensorboard = pl_loggers.TensorBoardLogger()
    trainer = Trainer(logger=tensorboard)

then access its API directly

.. code-block:: python

    def training_step(self):
        tensorboard = self.logger.experiment
        tensorboard.add_image()
        tensorboard.add_histogram(...)
        tensorboard.add_figure(...)

----

Comet.ml 
========
A

----

MLflow
======
A

----

Neptune.ai
==========
A

----

Tensorboard
===========
A

----

Weights and Biases
==================
A

----

Use multiple loggers
====================
A

----

****************************************
Track multiple metrics in the same chart
****************************************

To plot multiple metrics on one chart, pass in a dictionary

.. code-block:: python

    self.log("performance", {"acc": acc, "recall": recall})

----

*********************
Track hyperparameters
*********************
A

----

********************
Track model topology
********************
A