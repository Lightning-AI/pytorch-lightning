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
    F1,
    FBeta,
    HammingDistance,
    IoU,
    MetricCollection,
    Precision,
    PrecisionRecallCurve,
    Recall,
    ROC,
    StatScores,
)
from pytorch_lightning.metrics.functional import (
    auc,
    auroc,
    average_precision,
    confusion_matrix,
    f1,
    fbeta,
    hamming_distance,
    iou,
    precision,
    precision_recall,
    precision_recall_curve,
    recall,
    roc,
    stat_scores,
)
from pytorch_lightning.metrics.functional.accuracy import accuracy
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

    MetricCollection.__init__.warned = False
    with pytest.deprecated_call(
        match="`pytorch_lightning.metrics.metric.MetricCollection` was deprecated since v1.3.0 in favor"
        " of `torchmetrics.collections.MetricCollection`. It will be removed in v1.5.0."
    ):
        metrics = MetricCollection([Accuracy()])
    assert metrics(preds, target) == {'Accuracy': torch.tensor(0.1250)}


def test_v1_5_metric_accuracy():
    accuracy.warned = False

    preds = torch.tensor([0, 0, 1, 0, 1])
    target = torch.tensor([0, 0, 1, 1, 1])
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert accuracy(preds, target) == torch.tensor(0.8)

    Accuracy.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        Accuracy()


def test_v1_5_metric_auc_auroc():
    AUC.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        AUC()

    ROC.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        ROC()

    AUROC.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        AUROC()

    x = torch.tensor([0, 1, 2, 3])
    y = torch.tensor([0, 1, 2, 2])
    auc.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert auc(x, y) == torch.tensor(4.)

    preds = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 1, 1])
    roc.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        fpr, tpr, thrs = roc(preds, target, pos_label=1)
        assert torch.equal(fpr, torch.tensor([0., 0., 0., 0., 1.]))
        assert torch.allclose(tpr, torch.tensor([0.0000, 0.3333, 0.6667, 1.0000, 1.0000]), atol=1e-4)
        assert torch.equal(thrs, torch.tensor([4, 3, 2, 1, 0]))

    preds = torch.tensor([0.13, 0.26, 0.08, 0.19, 0.34])
    target = torch.tensor([0, 0, 1, 1, 1])
    auroc.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert auroc(preds, target) == torch.tensor(0.5)


def test_v1_5_metric_precision_recall():
    AveragePrecision.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        AveragePrecision()

    Precision.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        Precision()

    Recall.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        Recall()

    PrecisionRecallCurve.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        PrecisionRecallCurve()

    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 1, 1])
    average_precision.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert average_precision(pred, target) == torch.tensor(1.)

    precision.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert precision(pred, target) == torch.tensor(0.5)

    recall.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert recall(pred, target) == torch.tensor(0.5)

    precision_recall.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        prec, rc = precision_recall(pred, target)
        assert prec == torch.tensor(0.5)
        assert rc == torch.tensor(0.5)

    precision_recall_curve.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        prec, rc, thrs = precision_recall_curve(pred, target)
        assert torch.equal(prec, torch.tensor([1., 1., 1., 1.]))
        assert torch.allclose(rc, torch.tensor([1., 0.6667, 0.3333, 0.]), atol=1e-4)
        assert torch.equal(thrs, torch.tensor([1, 2, 3]))


def test_v1_5_metric_classif_mix():
    ConfusionMatrix.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        ConfusionMatrix(num_classes=1)

    FBeta.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        FBeta(num_classes=1)

    F1.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        F1(num_classes=1)

    HammingDistance.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        HammingDistance()

    StatScores.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        StatScores()

    target = torch.tensor([1, 1, 0, 0])
    preds = torch.tensor([0, 1, 0, 0])
    confusion_matrix.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert torch.equal(confusion_matrix(preds, target, num_classes=2), torch.tensor([[2., 0.], [1., 1.]]))

    target = torch.tensor([0, 1, 2, 0, 1, 2])
    preds = torch.tensor([0, 2, 1, 0, 0, 1])
    fbeta.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert torch.allclose(fbeta(preds, target, num_classes=3, beta=0.5), torch.tensor(0.3333), atol=1e-4)

    f1.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert torch.allclose(f1(preds, target, num_classes=3), torch.tensor(0.3333), atol=1e-4)

    target = torch.tensor([[0, 1], [1, 1]])
    preds = torch.tensor([[0, 1], [0, 1]])
    hamming_distance.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert hamming_distance(preds, target) == torch.tensor(0.25)

    preds = torch.tensor([1, 0, 2, 1])
    target = torch.tensor([1, 1, 2, 0])
    stat_scores.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert torch.equal(stat_scores(preds, target, reduce='micro'), torch.tensor([2, 2, 6, 2, 4]))


def test_v1_5_metric_detect():
    IoU.__init__.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        IoU(num_classes=1)

    target = torch.randint(0, 2, (10, 25, 25))
    pred = torch.tensor(target)
    pred[2:5, 7:13, 9:15] = 1 - pred[2:5, 7:13, 9:15]
    iou.warned = False
    with pytest.deprecated_call(match='It will be removed in v1.5.0'):
        assert torch.allclose(iou(pred, target), torch.tensor(0.9660), atol=1e-4)
