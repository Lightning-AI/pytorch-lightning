import numbers
from collections import Mapping, Sequence

import numpy as np
import pytest
import torch
from sklearn.metrics import accuracy_score, average_precision_score, auc

from pytorch_lightning.metrics.converters import _convert_to_numpy
from pytorch_lightning.metrics.sklearn import Accuracy, AveragePrecision, AUC
from pytorch_lightning.utilities.apply_func import apply_to_collection


@pytest.mark.parametrize(['metric_class', 'sklearn_func', 'inputs'], [
    pytest.param(Accuracy(), accuracy_score,
                 {'y_pred': torch.randint(low=0, high=10, size=(10,)),
                  'y_true': torch.randint(low=0, high=10, size=(10,))}),
    pytest.param(AUC(), auc, {'x': torch.arange(10, dtype=torch.float)/10,
                              'y': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2,
                                                 0.2, 0.3, 0.5, 0.6, 0.7])})
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
