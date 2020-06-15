Metrics
=======

Metrics are used to monitor model performance.

In this package we provide two major pieces of functionality.

    1. A Metric class you can use to implement metrics with built-in distributed (ddp) support which are device agnostic.
    2. A collection of popular metrics already implemented for you.

Example:

.. code-block:: python

    from pytorch_lightning.metrics.functional import accuracy

    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 2])

    # calculates accuracy across all GPUs and all Nodes used in training
    accuracy(pred, target)

    # tensor(0.9167)


--------------

Implement a metric
------------------
You can implement metrics as either a PyTorch metric or a Numpy metric. Numpy metrics
will slow down training, use PyTorch metrics when possible.

Use :class:`TensorMetric` to implement native PyTorch metrics. This class
handles automated DDP syncing and converts all inputs and outputs to tensors.

Use :class:`NumpyMetric` to implement numpy metrics. This class
handles automated DDP syncing and converts all inputs and outputs to tensors.

.. warning:: Numpy metrics might slow down your training substantially,
    since every metric computation requires a GPU sync to convert tensors to numpy.

TensorMetric
^^^^^^^^^^^^
Here's an example showing how to implement a TensorMetric

.. code-block:: python

    class RMSE(TensorMetric):
        def forward(self, x, y):
            return torch.sqrt(torch.mean(torch.pow(x-y, 2.0)))

.. autoclass:: pytorch_lightning.metrics.metric.TensorMetric
    :noindex:

NumpyMetric
^^^^^^^^^^^
Here's an example showing how to implement a NumpyMetric

.. code-block:: python

    class RMSE(NumpyMetric):
        def forward(self, x, y):
            return np.sqrt(np.mean(np.power(x-y, 2.0)))
        

.. autoclass:: pytorch_lightning.metrics.metric.NumpyMetric
    :noindex:

Class Metrics
-------------
The following are metrics which can be instantiated as part of a module definition (even with just
plain PyTorch).

.. code-block:: python

    from pytorch_lightning.metrics import Accuracy

    # Plain PyTorch
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.rmse = RMSE()

        def forward(self, x, y):
            y_hat = # ...
            acc = self.rmse(y_hat, y)

    # PyTorch Lightning
    class MyModule(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.rmse = RMSE()

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = # ...
            acc = self.rmse(y_hat, y)

These metrics even work when using distributed training:

.. code-block:: python

    model = MyModule()
    trainer = Trainer(gpus=8, num_nodes=2)

    # any metric automatically reduces across GPUs (even the ones you implement using Lightning)
    trainer.fit(model)

Accuracy
^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Accuracy
        :noindex:

AveragePrecision
^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.AveragePrecision
        :noindex:

AUROC
^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.AUROC
        :noindex:

ConfusionMatrix
^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.ConfusionMatrix
        :noindex:

DiceCoefficient
^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.DiceCoefficient
        :noindex:

F1
^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.F1
        :noindex:

FBeta
^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.FBeta
        :noindex:

PrecisionRecall
^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.PrecisionRecall
        :noindex:

Precision
^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Precision
        :noindex:

Recall
^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Recall
        :noindex:

ROC
^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.ROC
        :noindex:

MulticlassROC
^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.MulticlassROC
        :noindex:

MulticlassPrecisionRecall
^^^^^^^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.MulticlassPrecisionRecall
        :noindex:

Functional Metrics
------------------

accuracy (F)
^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.accuracy
        :noindex:

auc (F)
^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.auc
        :noindex:

auroc (F)
^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.auroc
        :noindex:

average_precision (F)
^^^^^^^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.average_precision
        :noindex:

confusion_matrix (F)
^^^^^^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.confusion_matrix
        :noindex:

dice_score (F)
^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.dice_score
        :noindex:

f1_score (F)
^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.f1_score
        :noindex:

fbeta_score (F)
^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.fbeta_score
        :noindex:

multiclass_precision_recall_curve (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.multiclass_precision_recall_curve
        :noindex:

multiclass_roc (F)
^^^^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.multiclass_roc
        :noindex:

precision (F)
^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.precision
        :noindex:

precision_recall (F)
^^^^^^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.precision_recall
        :noindex:

precision_recall_curve (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.precision_recall_curve
        :noindex:

recall (F)
^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.recall
        :noindex:

roc (F)
^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.roc
        :noindex:

stat_scores (F)
^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.stat_scores
        :noindex:

stat_scores_multiple_classes (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.stat_scores_multiple_classes
        :noindex:

Metric pre-processing
---------------------
Metric

to_categorical (F)
^^^^^^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.to_categorical
        :noindex:

to_onehot (F)
^^^^^^^^^^^^^
Metric

    .. autofunction:: pytorch_lightning.metrics.functional.to_onehot
        :noindex:
