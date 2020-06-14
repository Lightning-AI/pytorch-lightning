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
For native PyTorch implementations of metrics, it is recommended to use
the :class:`TensorMetric` which handles automated DDP syncing and conversions
to tensors for all inputs and outputs.

If your metrics implementation works on numpy, just use the
:class:`NumpyMetric`, which handles the automated conversion of
inputs to and outputs from numpy as well as automated ddp syncing.

.. warning:: Employing numpy in your metric calculation might slow
    down your training substantially, since every metric computation
    requires a GPU sync to convert tensors to numpy.

Metric
^^^^^^
asdf

    .. autoclass:: pytorch_lightning.metrics.metric.Metric
        :noindex:

TensorMetric
^^^^^^^^^^^^
asd

    .. autoclass:: pytorch_lightning.metrics.metric.TensorMetric
        :noindex:

NumpyMetric
^^^^^^^^^^^
asd

    .. autoclass:: pytorch_lightning.metrics.metric.NumpyMetric
        :noindex:

Class Metrics
-------------

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

