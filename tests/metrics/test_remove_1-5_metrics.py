# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test deprecated functionality which will be removed in v1.5.0"""

import pytest
import torch

from pytorch_lightning.metrics import (
    Accuracy,
    AUC,
    AUROC,
    AveragePrecision,
    ConfusionMatrix,
    ExplainedVariance,
    F1,
    FBeta,
    HammingDistance,
    IoU,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    PSNR,
    R2Score,
    Recall,
    ROC,
    SSIM,
    StatScores,
)
from pytorch_lightning.metrics.functional import (
    auc,
    auroc,
    average_precision,
    bleu_score,
    confusion_matrix,
    embedding_similarity,
    explained_variance,
    f1,
    fbeta,
    hamming_distance,
    iou,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    precision,
    precision_recall,
    precision_recall_curve,
    psnr,
    r2score,
    recall,
    roc,
    ssim,
    stat_scores,
)
from pytorch_lightning.metrics.functional.accuracy import accuracy
from pytorch_lightning.metrics.functional.mean_relative_error import mean_relative_error
from pytorch_lightning.metrics.utils import get_num_classes, select_topk, to_categorical, to_onehot


def test_v1_5_metrics_utils():
    x = torch.tensor([1, 2, 3])
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.equal(to_onehot(x), torch.Tensor([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).to(int))

    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert get_num_classes(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 0])) == 4

    x = torch.tensor([[1.1, 2.0, 3.0], [2.0, 1.0, 0.5]])
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.equal(select_topk(x, topk=2), torch.Tensor([[0, 1, 1], [1, 1, 0]]).to(torch.int32))

    x = torch.tensor([[0.2, 0.5], [0.9, 0.1]])
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.equal(to_categorical(x), torch.Tensor([1, 0]).to(int))


def test_v1_5_metrics_collection():
    target = torch.tensor([0, 2, 0, 2, 0, 1, 0, 2])
    preds = torch.tensor([2, 1, 2, 0, 1, 2, 2, 2])

    MetricCollection.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0."):
        metrics = MetricCollection([Accuracy()])
    assert metrics(preds, target) == {"Accuracy": torch.tensor(0.1250)}


def test_v1_5_metric_accuracy():
    accuracy._warned = False

    preds = torch.tensor([0, 0, 1, 0, 1])
    target = torch.tensor([0, 0, 1, 1, 1])
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert accuracy(preds, target) == torch.tensor(0.8)

    Accuracy.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        Accuracy()


def test_v1_5_metric_auc_auroc():
    AUC.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        AUC()

    ROC.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        ROC()

    AUROC.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        AUROC()

    x = torch.tensor([0, 1, 2, 3])
    y = torch.tensor([0, 1, 2, 2])
    auc._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert auc(x, y) == torch.tensor(4.0)

    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 1, 1])
    roc._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        fpr, tpr, thrs = roc(preds, target, pos_label=1)
    assert torch.equal(fpr, torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(tpr, torch.tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000]), atol=1e-4)
    assert torch.equal(thrs, torch.tensor([4, 3, 2, 1, 0]))

    preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = torch.tensor([0, 0, 1, 1, 1])
    auroc._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert auroc(preds, target) == torch.tensor(0.5)


def test_v1_5_metric_precision_recall():
    AveragePrecision.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        AveragePrecision()

    Precision.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        Precision()

    Recall.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        Recall()

    PrecisionRecallCurve.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        PrecisionRecallCurve()

    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 1, 1])
    average_precision._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert average_precision(pred, target) == torch.tensor(1.0)

    precision._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert precision(pred, target) == torch.tensor(0.5)

    recall._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert recall(pred, target) == torch.tensor(0.5)

    precision_recall._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        prec, rc = precision_recall(pred, target)
        assert prec == torch.tensor(0.5)
        assert rc == torch.tensor(0.5)

    precision_recall_curve._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        prec, rc, thrs = precision_recall_curve(pred, target)
    assert torch.equal(prec, torch.tensor([1.0, 1.0, 1.0, 1.0]))
    assert torch.allclose(rc, torch.tensor([1.0, 0.6667, 0.3333, 0.0]), atol=1e-4)
    assert torch.equal(thrs, torch.tensor([1, 2, 3]))


def test_v1_5_metric_classif_mix():
    ConfusionMatrix.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        ConfusionMatrix(num_classes=1)

    FBeta.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        FBeta(num_classes=1)

    F1.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        F1(num_classes=1)

    HammingDistance.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        HammingDistance()

    StatScores.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        StatScores()

    target = torch.tensor([1, 1, 0, 0])
    preds = torch.tensor([0, 1, 0, 0])
    confusion_matrix._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.equal(
            confusion_matrix(preds, target, num_classes=2).float(), torch.tensor([[2.0, 0.0], [1.0, 1.0]])
        )

    target = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = torch.tensor([0, 2, 1, 0, 0, 1])
    fbeta._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.allclose(fbeta(preds, target, num_classes=3, beta=0.5), torch.tensor(0.3333), atol=1e-4)

    f1._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.allclose(f1(preds, target, num_classes=3), torch.tensor(0.3333), atol=1e-4)

    target = torch.tensor([[0, 1], [1, 1]])
    preds = torch.tensor([[0, 1], [0, 1]])
    hamming_distance._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert hamming_distance(preds, target) == torch.tensor(0.25)

    preds = torch.tensor([1, 0, 2, 1])
    target = torch.tensor([1, 1, 2, 0])
    stat_scores._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert torch.equal(stat_scores(preds, target, reduce="micro"), torch.tensor([2, 2, 6, 2, 4]))


def test_v1_5_metric_detect():
    IoU.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        IoU(num_classes=1)

    target = torch.randint(0, 2, (10, 25, 25))
    preds = torch.tensor(target)
    preds[2:5, 7:13, 9:15] = 1 - preds[2:5, 7:13, 9:15]
    iou._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        res = iou(preds, target)
    assert torch.allclose(res, torch.tensor(0.9660), atol=1e-4)


def test_v1_5_metric_regress():
    ExplainedVariance.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        ExplainedVariance()

    MeanAbsoluteError.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        MeanAbsoluteError()

    MeanSquaredError.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        MeanSquaredError()

    MeanSquaredLogError.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        MeanSquaredLogError()

    target = torch.tensor([3, -0.5, 2, 7])
    preds = torch.tensor([2.5, 0.0, 2, 8])
    explained_variance._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        res = explained_variance(preds, target)
    assert torch.allclose(res, torch.tensor(0.9572), atol=1e-4)

    x = torch.tensor([0.0, 1, 2, 3])
    y = torch.tensor([0.0, 1, 2, 2])
    mean_absolute_error._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert mean_absolute_error(x, y) == 0.25

    mean_relative_error._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert mean_relative_error(x, y) == 0.125

    mean_squared_error._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        assert mean_squared_error(x, y) == 0.25

    mean_squared_log_error._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        res = mean_squared_log_error(x, y)
    assert torch.allclose(res, torch.tensor(0.0207), atol=1e-4)

    PSNR.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        PSNR()

    R2Score.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        R2Score()

    SSIM.__init__._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        SSIM()

    preds = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    target = torch.tensor([[3.0, 2.0], [1.0, 0.0]])
    psnr._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        res = psnr(preds, target)
    assert torch.allclose(res, torch.tensor(2.5527), atol=1e-4)

    target = torch.tensor([3, -0.5, 2, 7])
    preds = torch.tensor([2.5, 0.0, 2, 8])
    r2score._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        res = r2score(preds, target)
    assert torch.allclose(res, torch.tensor(0.9486), atol=1e-4)

    preds = torch.rand([16, 1, 16, 16])
    target = preds * 0.75
    ssim._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        res = ssim(preds, target)
    assert torch.allclose(res, torch.tensor(0.9219), atol=1e-4)


def test_v1_5_metric_others():
    translate_corpus = ["the cat is on the mat".split()]
    reference_corpus = [["there is a cat on the mat".split(), "a cat is on the mat".split()]]
    bleu_score._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        res = bleu_score(translate_corpus, reference_corpus)
    assert torch.allclose(res, torch.tensor(0.7598), atol=1e-4)

    embeddings = torch.tensor([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0]])
    embedding_similarity._warned = False
    with pytest.deprecated_call(match="It will be removed in v1.5.0"):
        res = embedding_similarity(embeddings)
    assert torch.allclose(
        res, torch.tensor([[0.0000, 1.0000, 0.9759], [1.0000, 0.0000, 0.9759], [0.9759, 0.9759, 0.0000]]), atol=1e-4
    )
