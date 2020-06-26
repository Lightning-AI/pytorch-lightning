import numbers
from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    precision_score as sk_precision,
    recall_score as sk_recall,
    f1_score as sk_f1_score,
    fbeta_score as sk_fbeta_score,
    confusion_matrix as sk_confusion_matrix,
    average_precision_score as sk_average_precision,
    auc as sk_auc,
    precision_recall_curve as sk_precision_recall_curve,
    roc_curve as sk_roc_curve,
    roc_auc_score as sk_roc_auc_score,
)

from pytorch_lightning.metrics.converters import _convert_to_numpy
from pytorch_lightning.metrics.sklearns import (
    Accuracy,
    AveragePrecision,
    AUC,
    ConfusionMatrix,
    F1,
    FBeta,
    Precision,
    Recall,
    PrecisionRecallCurve,
    ROC,
    AUROC
)
from pytorch_lightning.utilities.apply_func import apply_to_collection


def _xy_only(func):
    def new_func(*args, **kwargs):
        return np.array(func(*args, **kwargs)[:2])
    return new_func


@pytest.mark.parametrize(['metric_class', 'sklearn_func', 'inputs'], [
    pytest.param(Accuracy(), sk_accuracy,
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='Accuracy'),
    pytest.param(AUC(), sk_auc,
                 {'x': torch.arange(10, dtype=torch.float) / 10,
                  'y': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.6, 0.7])},
                 id='AUC'),
    pytest.param(AveragePrecision(), sk_average_precision,
                 {'y_score': torch.randint(2, size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='AveragePrecision'),
    pytest.param(ConfusionMatrix(), sk_confusion_matrix,
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='ConfusionMatrix'),
    pytest.param(F1(average='macro'), partial(sk_f1_score, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='F1'),
    pytest.param(FBeta(beta=0.5, average='macro'), partial(sk_fbeta_score, beta=0.5, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='FBeta'),
    pytest.param(Precision(average='macro'), partial(sk_precision, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='Precision'),
    pytest.param(Recall(average='macro'), partial(sk_recall, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='Recall'),
    pytest.param(PrecisionRecallCurve(), _xy_only(sk_precision_recall_curve),
                 {'probas_pred': torch.rand(size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='PrecisionRecallCurve'),
    pytest.param(ROC(), _xy_only(sk_roc_curve),
                 {'y_score': torch.rand(size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='ROC'),
    pytest.param(AUROC(), sk_roc_auc_score,
                 {'y_score': torch.rand(size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='AUROC'),
])
def test_sklearn_metric(metric_class, sklearn_func, inputs):
    numpy_inputs = apply_to_collection(inputs, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)

    sklearn_result = sklearn_func(**numpy_inputs)
    lightning_result = metric_class(**inputs)
    assert np.allclose(sklearn_result, lightning_result, atol=1e-5)

    sklearn_result = apply_to_collection(
        sklearn_result, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)

    lightning_result = apply_to_collection(
        lightning_result, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)

    assert np.allclose(sklearn_result, lightning_result, atol=1e-5)
    assert isinstance(lightning_result, type(sklearn_result))
