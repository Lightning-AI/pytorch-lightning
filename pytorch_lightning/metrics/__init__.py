"""
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


"""

from pytorch_lightning.metrics.metric import Metric, TensorMetric, NumpyMetric
from pytorch_lightning.metrics.sklearn import (
    SklearnMetric, Accuracy, AveragePrecision, AUC, ConfusionMatrix, F1, FBeta,
    Precision, Recall, PrecisionRecallCurve, ROC, AUROC)
from pytorch_lightning.metrics.converters import numpy_metric, tensor_metric
