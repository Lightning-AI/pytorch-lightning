Metrics
=======

Metrics are generally used to monitor model performance.

The following package aims to provide the most convenient ones as well
as a structure to implement your custom metrics for all the fancy research
you want to do.

For native PyTorch implementations of metrics, it is recommended to use
the :class:`TensorMetric` which handles automated DDP syncing and conversions
to tensors for all inputs and outputs.

If your metrics implementation works on numpy, just use the
:class:`NumpyMetric`, which handles the automated conversion of
inputs to and outputs from numpy as well as automated ddp syncing.

.. warning:: Employing numpy in your metric calculation might slow
    down your training substantially, since every metric computation
    requires a GPU sync to convert tensors to numpy.


Implement a metric
------------------
.. role:: hidden
    :class: hidden-section

asdfasd

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

SKLearn Metrics
---------------

SklearnMetric
^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.SklearnMetric
        :noindex:

Accuracy
^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.Accuracy
            :noindex:

AUC
^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.AUC
            :noindex:

AUROC
^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.AUROC
            :noindex:

AveragePrecision
^^^^^^^^^^^^^^^^
Metric


    .. autoclass:: pytorch_lightning.metrics.sklearn.AveragePrecision
            :noindex:


ConfusionMatrix
^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.ConfusionMatrix
            :noindex:

F1
^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.F1
            :noindex:

FBeta
^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.FBeta
            :noindex:

Precision
^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.Precision
            :noindex:

Recall
^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.Recall
            :noindex:

PrecisionRecallCurve
^^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.PrecisionRecallCurve
            :noindex:

ROC
^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.sklearn.ROC
            :noindex:

PyTorch Metrics
---------------

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

