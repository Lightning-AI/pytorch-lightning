from pytorch_lightning.metrics.classification import (
    Accuracy,
    AveragePrecision,
    ConfusionMatrix,
    F1,
    FBeta,
    Recall,
    ROC,
    AUROC,
    DiceCoefficient,
    MulticlassPrecisionRecall,
    MulticlassROC,
    Precision,
    PrecisionRecall,
    IoU,
)
from pytorch_lightning.metrics.converters import numpy_metric, tensor_metric
from pytorch_lightning.metrics.metric import Metric, TensorMetric, NumpyMetric
from pytorch_lightning.metrics.nlp import BLEUScore
from pytorch_lightning.metrics.regression import (
    MAE,
    MSE,
    PSNR,
    RMSE,
    RMSLE,
    SSIM
)
from pytorch_lightning.metrics.sklearns import (
    AUC,
    PrecisionRecallCurve,
    SklearnMetric,
)

__classification_metrics = [
    "AUC",
    "AUROC",
    "Accuracy",
    "AveragePrecision",
    "ConfusionMatrix",
    "DiceCoefficient",
    "F1",
    "FBeta",
    "MulticlassPrecisionRecall",
    "MulticlassROC",
    "Precision",
    "PrecisionRecall",
    "PrecisionRecallCurve",
    "ROC",
    "Recall",
    "IoU",
]
__regression_metrics = [
    "MAE",
    "MSE",
    "PSNR",
    "RMSE",
    "RMSLE",
    "SSIM"
]
__sequence_metrics = ["BLEUScore"]
__all__ = __regression_metrics + __classification_metrics + ["SklearnMetric"] + __sequence_metrics
