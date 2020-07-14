.. testsetup:: *

    import torch
    from torch.nn import Module
    from pytorch_lightning.core.lightning import LightningModule
    from pytorch_lightning.metrics import TensorMetric, NumpyMetric

Metrics
=======
This is a general package for PyTorch Metrics. These can also be used with regular non-lightning PyTorch code.
Metrics are used to monitor model performance.

In this package, we provide two major pieces of functionality.

1. A Metric class you can use to implement metrics with built-in distributed (ddp) support which are device agnostic.
2. A collection of ready to use popular metrics. There are two types of metrics: Class metrics and Functional metrics.
3. An interface to call `sklearns metrics <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_

Example::

    from pytorch_lightning.metrics.functional import accuracy

    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 2])

    # calculates accuracy across all GPUs and all Nodes used in training
    accuracy(pred, target)

.. warning::
    The metrics package is still in development! If we're missing a metric or you find a mistake, please send a PR!
    to a few metrics. Please feel free to create an issue/PR if you have a proposed  metric or have found a bug.

----------------

Implement a metric
------------------
You can implement metrics as either a PyTorch metric or a Numpy metric (It is recommended to use PyTorch metrics when possible,
since Numpy metrics slow down training).

Use :class:`TensorMetric` to implement native PyTorch metrics. This class
handles automated DDP syncing and converts all inputs and outputs to tensors.

Use :class:`NumpyMetric` to implement numpy metrics. This class
handles automated DDP syncing and converts all inputs and outputs to tensors.

.. warning::
    Numpy metrics might slow down your training substantially,
    since every metric computation requires a GPU sync to convert tensors to numpy.

----------------

TensorMetric
^^^^^^^^^^^^
Here's an example showing how to implement a TensorMetric

.. testcode::

    class RMSE(TensorMetric):
        def forward(self, x, y):
            return torch.sqrt(torch.mean(torch.pow(x-y, 2.0)))

.. autoclass:: pytorch_lightning.metrics.metric.TensorMetric
    :noindex:

----------------

NumpyMetric
^^^^^^^^^^^
Here's an example showing how to implement a NumpyMetric

.. testcode::

    class RMSE(NumpyMetric):
        def forward(self, x, y):
            return np.sqrt(np.mean(np.power(x-y, 2.0)))
        

.. autoclass:: pytorch_lightning.metrics.metric.NumpyMetric
    :noindex:

----------------

Class Metrics
-------------
Class metrics can be instantiated as part of a module definition (even with just
plain PyTorch).

.. testcode::

    from pytorch_lightning.metrics import Accuracy

    # Plain PyTorch
    class MyModule(Module):
        def __init__(self):
            super().__init__()
            self.metric = Accuracy()

        def forward(self, x, y):
            y_hat = ...
            acc = self.metric(y_hat, y)

    # PyTorch Lightning
    class MyModule(LightningModule):
        def __init__(self):
            super().__init__()
            self.metric = Accuracy()

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = ...
            acc = self.metric(y_hat, y)

These metrics even work when using distributed training:

.. code-block:: python

    model = MyModule()
    trainer = Trainer(gpus=8, num_nodes=2)

    # any metric automatically reduces across GPUs (even the ones you implement using Lightning)
    trainer.fit(model)

Accuracy
^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.Accuracy
    :noindex:

AveragePrecision
^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.AveragePrecision
    :noindex:

AUROC
^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.AUROC
    :noindex:

ConfusionMatrix
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.ConfusionMatrix
    :noindex:

DiceCoefficient
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.DiceCoefficient
    :noindex:

F1
^^

.. autoclass:: pytorch_lightning.metrics.classification.F1
    :noindex:

FBeta
^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.FBeta
    :noindex:

PrecisionRecall
^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.PrecisionRecall
    :noindex:

Precision
^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.Precision
    :noindex:

Recall
^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.Recall
    :noindex:

ROC
^^^

.. autoclass:: pytorch_lightning.metrics.classification.ROC
    :noindex:

MAE
^^^

.. autoclass:: pytorch_lightning.metrics.regression.MAE
    :noindex:

MSE
^^^

.. autoclass:: pytorch_lightning.metrics.regression.MSE
    :noindex:

MulticlassROC
^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.MulticlassROC
    :noindex:

MulticlassPrecisionRecall
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: pytorch_lightning.metrics.classification.MulticlassPrecisionRecall
    :noindex:

IoU
^^^

.. autoclass:: pytorch_lightning.metrics.classification.IoU
    :noindex:

RMSE
^^^^

.. autoclass:: pytorch_lightning.metrics.regression.RMSE
    :noindex:

RMSLE
^^^^^

.. autoclass:: pytorch_lightning.metrics.regression.RMSLE
    :noindex:

Bleu
^^^^^

.. autoclass:: pytorch_lightning.metrics.seq2seq.Bleu
    :noindex:

----------------

Functional Metrics
------------------
Functional metrics can be called anywhere (even used with just plain PyTorch).

.. code-block:: python

    from pytorch_lightning.metrics.functional import accuracy

    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 2])

    # calculates accuracy across all GPUs and all Nodes used in training
    accuracy(pred, target)

These metrics even work when using distributed training:

.. code-block:: python

    class MyModule(...):
        def forward(self, x, y):
            return accuracy(x, y)

    model = MyModule()
    trainer = Trainer(gpus=8, num_nodes=2)

    # any metric automatically reduces across GPUs (even the ones you implement using Lightning)
    trainer.fit(model)


accuracy (F)
^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.accuracy
    :noindex:

auc (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.auc
    :noindex:

auroc (F)
^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.auroc
    :noindex:

average_precision (F)
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.average_precision
    :noindex:

confusion_matrix (F)
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.confusion_matrix
    :noindex:

dice_score (F)
^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.dice_score
    :noindex:

f1_score (F)
^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.f1_score
    :noindex:

fbeta_score (F)
^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.fbeta_score
    :noindex:

multiclass_precision_recall_curve (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.multiclass_precision_recall_curve
    :noindex:

multiclass_roc (F)
^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.multiclass_roc
    :noindex:

precision (F)
^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.precision
    :noindex:

precision_recall (F)
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.precision_recall
    :noindex:

precision_recall_curve (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.precision_recall_curve
    :noindex:

recall (F)
^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.recall
    :noindex:

roc (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.roc
    :noindex:

stat_scores (F)
^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.stat_scores
    :noindex:

iou (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.iou
    :noindex:

mse (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.mse
    :noindex:

rmse (F)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.rmse
    :noindex:

mae (F)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.mae
    :noindex:

rmsle (F)
^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.rmsle
    :noindex:

psnr (F)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.psnr
    :noindex:

stat_scores_multiple_classes (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.stat_scores_multiple_classes
    :noindex:

----------------

Metric pre-processing
---------------------

to_categorical (F)
^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.to_categorical
    :noindex:

to_onehot (F)
^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.functional.to_onehot
    :noindex:

----------------

Sklearn interface
-----------------
    
Lightning supports `sklearns metrics module <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics>`_ 
as a backend for calculating metrics. Sklearns metrics are well tested and robust, 
but requires conversion between pytorch and numpy thus may slow down your computations.

To use the sklearn backend of metrics simply import as

.. code-block:: python
    
    import pytorch_lightning.metrics.sklearns import plm
    metric = plm.Accuracy(normalize=True)
    val = metric(pred, target)
    
Each converted sklearn metric comes has the same interface as its 
original counterpart (e.g. accuracy takes the additional `normalize` keyword). 
Like the native Lightning metrics, these converted sklearn metrics also come 
with built-in distributed (ddp) support.

SklearnMetric (sk)
^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.SklearnMetric
    :noindex:

Accuracy (sk)
^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Accuracy
    :noindex:

AUC (sk)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.AUC
    :noindex:

AveragePrecision (sk)
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.AveragePrecision
    :noindex:

    
ConfusionMatrix (sk)
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.ConfusionMatrix
    :noindex:

F1 (sk)
^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.F1
    :noindex:

FBeta (sk)
^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.FBeta
    :noindex:

Precision (sk)
^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Precision
    :noindex:

Recall (sk)
^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.Recall
    :noindex:

PrecisionRecallCurve (sk)
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.PrecisionRecallCurve
    :noindex:

ROC (sk)
^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.ROC
    :noindex:

AUROC (sk)
^^^^^^^^^^

.. autofunction:: pytorch_lightning.metrics.sklearns.AUROC
    :noindex:
