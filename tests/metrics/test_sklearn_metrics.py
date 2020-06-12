import numbers
from collections import Mapping, Sequence
from functools import partial

import numpy as np
import pytest
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    auc,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score
)

from pytorch_lightning.metrics.converters import _convert_to_numpy
from pytorch_lightning.metrics.sklearn import (
    Accuracy, AveragePrecision, AUC, ConfusionMatrix, F1, FBeta,
    Precision, Recall, PrecisionRecallCurve, ROC, AUROC)
from pytorch_lightning.utilities.apply_func import apply_to_collection


def xy_only(func):
    def new_func(*args, **kwargs):
        return np.array(func(*args, **kwargs)[:2])

    return new_func


@pytest.mark.parametrize(['metric_class', 'sklearn_func', 'inputs'], [
    pytest.param(Accuracy(), accuracy_score,
                 {'y_pred': torch.randint(low=0, high=10, size=(128,)),
                  'y_true': torch.randint(low=0, high=10, size=(128,))},
                 id='Accuracy'),
    pytest.param(AUC(), auc, {'x': torch.arange(10, dtype=torch.float) / 10,
                              'y': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.6, 0.7])},
                 id='AUC'),
    pytest.param(AveragePrecision(), average_precision_score,
                 {'y_score': torch.randint(2, size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='AveragePrecision'),
    pytest.param(ConfusionMatrix(), confusion_matrix,
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='ConfusionMatrix'),
    pytest.param(F1(average='macro'), partial(f1_score, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='F1'),
    pytest.param(FBeta(beta=0.5, average='macro'), partial(fbeta_score, beta=0.5, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='FBeta'),
    pytest.param(Precision(average='macro'), partial(precision_score, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='Precision'),
    pytest.param(Recall(average='macro'), partial(recall_score, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='Recall'),
    pytest.param(PrecisionRecallCurve(), xy_only(precision_recall_curve),
                 {'probas_pred': torch.rand(size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='PrecisionRecallCurve'),
    pytest.param(ROC(), xy_only(roc_curve),
                 {'y_score': torch.rand(size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='ROC'),
    pytest.param(AUROC(), roc_auc_score,
                 {'y_score': torch.rand(size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='AUROC'),
])
def test_sklearn_metric(metric_class, sklearn_func, inputs: dict):
    numpy_inputs = apply_to_collection(
        inputs, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)

    sklearn_result = sklearn_func(**numpy_inputs)
    lightning_result = metric_class(**inputs)

    sklearn_result = apply_to_collection(
        sklearn_result, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)

    lightning_result = apply_to_collection(
        lightning_result, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)

    assert isinstance(lightning_result, type(sklearn_result))

    if isinstance(lightning_result, np.ndarray):
        assert np.allclose(lightning_result, sklearn_result)
    elif isinstance(lightning_result, Mapping):
        for key in lightning_result.keys():
            assert np.allclose(lightning_result[key], sklearn_result[key])

    elif isinstance(lightning_result, Sequence):
        for val_lightning, val_sklearn in zip(lightning_result, sklearn_result):
            assert np.allclose(val_lightning, val_sklearn)

    else:
        raise TypeError
