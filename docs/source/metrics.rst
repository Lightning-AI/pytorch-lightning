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

Metrics
-------

Accuracy (F)
^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Accuracy
            :noindex:

AveragePrecision (F)
^^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.AveragePrecision
            :noindex:

AUROC (F)
^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.AUROC
            :noindex:

ConfusionMatrix (F)
^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.ConfusionMatrix
            :noindex:

DiceCoefficient (F)
^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.DiceCoefficient
            :noindex:

F1 (F)
^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.F1
            :noindex:

FBeta (F)
^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.FBeta
            :noindex:

PrecisionRecall (F)
^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.PrecisionRecall
            :noindex:

Precision (F)
^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Precision
            :noindex:

Recall (F)
^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Recall
            :noindex:

ROC (F)
^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.ROC
            :noindex:

MulticlassROC (F)
^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.MulticlassROC
            :noindex:

MulticlassPrecisionRecall (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.MulticlassPrecisionRecall
            :noindex:

Functional Metrics
------------------

Accuracy (F)
^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Accuracy
            :noindex:

AveragePrecision (F)
^^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.AveragePrecision
            :noindex:

AUROC (F)
^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.AUROC
            :noindex:

ConfusionMatrix (F)
^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.ConfusionMatrix
            :noindex:

DiceCoefficient (F)
^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.DiceCoefficient
            :noindex:

F1 (F)
^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.F1
            :noindex:

FBeta (F)
^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.FBeta
            :noindex:

PrecisionRecall (F)
^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.PrecisionRecall
            :noindex:

Precision (F)
^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Precision
            :noindex:

Recall (F)
^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.Recall
            :noindex:

ROC (F)
^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.ROC
            :noindex:

MulticlassROC (F)
^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.MulticlassROC
            :noindex:

MulticlassPrecisionRecall (F)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Metric

    .. autoclass:: pytorch_lightning.metrics.classification.MulticlassPrecisionRecall
            :noindex:
