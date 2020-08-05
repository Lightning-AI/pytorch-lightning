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
    balanced_accuracy_score as sk_balanced_accuracy_score,
    dcg_score as sk_dcg_score,
    mean_absolute_error as sk_mean_absolute_error,
    mean_squared_error as sk_mean_squared_error,
    mean_squared_log_error as sk_mean_squared_log_error,
    median_absolute_error as sk_median_absolute_error,
    r2_score as sk_r2_score,
    mean_poisson_deviance as sk_mean_poisson_deviance,
    mean_gamma_deviance as sk_mean_gamma_deviance,
    mean_tweedie_deviance as sk_mean_tweedie_deviance,
    explained_variance_score as sk_explained_variance_score,
    cohen_kappa_score as sk_cohen_kappa_score,
    hamming_loss as sk_hamming_loss,
    hinge_loss as sk_hinge_loss,
    jaccard_score as sk_jaccard_score
)

from pytorch_lightning.metrics.converters import _convert_to_numpy
from pytorch_lightning.metrics.sklearns import (
    Accuracy,
    AUC,
    AveragePrecision,
    BalancedAccuracy,
    ConfusionMatrix,
    CohenKappaScore,
    DCG,
    F1,
    FBeta,
    Hamming,
    Hinge,
    Jaccard,
    Precision,
    Recall,
    PrecisionRecallCurve,
    ROC,
    AUROC,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    MedianAbsoluteError,
    R2Score,
    MeanPoissonDeviance,
    MeanGammaDeviance,
    MeanTweedieDeviance,
    ExplainedVariance,
)
from pytorch_lightning.utilities.apply_func import apply_to_collection


def _xy_only(func):
    def new_func(*args, **kwargs):
        return np.array(func(*args, **kwargs)[:2])
    return new_func


@pytest.mark.parametrize(['metric_class', 'sklearn_func', 'inputs'], [
    pytest.param(Accuracy(), sk_accuracy,
                 {'y_pred': torch.randint(10, size=(128,)),
                  'y_true': torch.randint(10, size=(128,))},
                 id='Accuracy'),
    pytest.param(AUC(), sk_auc,
                 {'x': torch.arange(10, dtype=torch.float) / 10,
                  'y': torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.6, 0.7])},
                 id='AUC'),
    pytest.param(AveragePrecision(), sk_average_precision,
                 {'y_score': torch.randint(2, size=(128,)),
                  'y_true': torch.randint(2, size=(128,))},
                 id='AveragePrecision'),
    pytest.param(ConfusionMatrix(), sk_confusion_matrix,
                 {'y_pred': torch.randint(10, size=(128,)),
                  'y_true': torch.randint(10, size=(128,))},
                 id='ConfusionMatrix'),
    pytest.param(F1(average='macro'), partial(sk_f1_score, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)),
                  'y_true': torch.randint(10, size=(128,))},
                 id='F1'),
    pytest.param(FBeta(beta=0.5, average='macro'), partial(sk_fbeta_score, beta=0.5, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)),
                  'y_true': torch.randint(10, size=(128,))},
                 id='FBeta'),
    pytest.param(Precision(average='macro'), partial(sk_precision, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)),
                  'y_true': torch.randint(10, size=(128,))},
                 id='Precision'),
    pytest.param(Recall(average='macro'), partial(sk_recall, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)),
                  'y_true': torch.randint(10, size=(128,))},
                 id='Recall'),
    pytest.param(PrecisionRecallCurve(), _xy_only(sk_precision_recall_curve),
                 {'probas_pred': torch.rand(size=(128,)),
                  'y_true': torch.randint(2, size=(128,))},
                 id='PrecisionRecallCurve'),
    pytest.param(ROC(), _xy_only(sk_roc_curve),
                 {'y_score': torch.rand(size=(128,)),
                  'y_true': torch.randint(2, size=(128,))},
                 id='ROC'),
    pytest.param(AUROC(), sk_roc_auc_score,
                 {'y_score': torch.rand(size=(128,)),
                  'y_true': torch.randint(2, size=(128,))},
                 id='AUROC'),
    pytest.param(BalancedAccuracy(), sk_balanced_accuracy_score,
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='BalancedAccuracy'),
    pytest.param(DCG(), sk_dcg_score,
                 {'y_score': torch.rand(size=(128, 3)), 'y_true': torch.randint(3, size=(128, 3))},
                 id='DCG'),
    pytest.param(ExplainedVariance(), sk_explained_variance_score,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='ExplainedVariance'),
    pytest.param(MeanAbsoluteError(), sk_mean_absolute_error,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='MeanAbsolutError'),
    pytest.param(MeanSquaredError(), sk_mean_squared_error,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='MeanSquaredError'),
    pytest.param(MeanSquaredLogError(), sk_mean_squared_log_error,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='MeanSquaredLogError'),
    pytest.param(MedianAbsoluteError(), sk_median_absolute_error,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='MedianAbsoluteError'),
    pytest.param(R2Score(), sk_r2_score,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='R2Score'),
    pytest.param(MeanPoissonDeviance(), sk_mean_poisson_deviance,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='MeanPoissonDeviance'),
    pytest.param(MeanGammaDeviance(), sk_mean_gamma_deviance,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='MeanGammaDeviance'),
    pytest.param(MeanTweedieDeviance(), sk_mean_tweedie_deviance,
                 {'y_pred': torch.rand(size=(128,)), 'y_true': torch.rand(size=(128,))},
                 id='MeanTweedieDeviance'),
    pytest.param(CohenKappaScore(), sk_cohen_kappa_score,
                 {'y1': torch.randint(3, size=(128,)), 'y2': torch.randint(3, size=(128,))},
                 id='CohenKappaScore'),
    pytest.param(Hamming(), sk_hamming_loss,
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='Hamming'),
    pytest.param(Hinge(), sk_hinge_loss,
                 {'pred_decision': torch.randn(size=(128,)), 'y_true': torch.randint(2, size=(128,))},
                 id='Hinge'),
    pytest.param(Jaccard(average='macro'), partial(sk_jaccard_score, average='macro'),
                 {'y_pred': torch.randint(10, size=(128,)), 'y_true': torch.randint(10, size=(128,))},
                 id='Jaccard')
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
