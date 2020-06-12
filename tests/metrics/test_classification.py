# NOTE: This file only tests if modules with arguments are running fine.
#   The actual metric implementation is tested in functional/test_classification.py
#   Especially reduction and reducing across processes won't be tested here!

import pytest

import torch

from pytorch_lightning.metrics.classification import *


@pytest.fixture
def random():
    torch.manual_seed(0)


@pytest.mark.parametrize(['num_classes'], [
    pytest.param(1),
    pytest.param(None)
])
def test_accuracy(num_classes):
    acc = Accuracy(num_classes=num_classes)

    assert acc.name == 'accuracy'

    result = acc(pred=torch.tensor([[0, 1, 1], [1, 0, 1]]),
                 target=torch.tensor([[0, 0, 1], [1, 0, 1]]))

    assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize(['normalize'], [
    pytest.param(False),
    pytest.param(True)
])
def test_confusion_matrix(normalize):
    conf_matrix = ConfusionMatrix(normalize=normalize)
    assert conf_matrix.name == 'confusion_matrix'

    target = (torch.arange(120) % 3).view(-1, 1)
    pred = target.clone()

    cm = conf_matrix(pred, target)

    assert isinstance(cm, torch.Tensor)


@pytest.mark.parametrize(['pos_label'], [
    pytest.param(1.),
    pytest.param(2.)
])
def test_precision_recall(pos_label):
    pred, target = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1])

    pr_curve = PrecisionRecall(pos_label=pos_label)
    assert pr_curve.name == 'precision_recall_curve'

    pr = pr_curve(pred=pred, target=target, sample_weight=[0.1, 0.2, 0.3, 0.4])

    assert isinstance(pr, tuple)
    assert len(pr) == 3
    for tmp in pr:
        assert isinstance(tmp, torch.Tensor)


@pytest.mark.parametrize(['num_classes'], [
    pytest.param(1),
    pytest.param(None)
])
def test_precision(num_classes):
    precision = Precision(num_classes=num_classes)

    assert precision.name == 'precision'
    pred, target = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1])

    prec = precision(pred=pred, target=target)

    assert isinstance(prec, torch.Tensor)


@pytest.mark.parametrize(['num_classes'], [
    pytest.param(1),
    pytest.param(None)
])
def test_recall(num_classes):
    recall = Recall(num_classes=num_classes)

    assert recall.name == 'recall'
    pred, target = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1])

    rec = recall(pred=pred, target=target)

    assert isinstance(rec, torch.Tensor)


@pytest.mark.parametrize(['pos_label'], [
    pytest.param(1.),
    pytest.param(2.)
])
def test_average_precision(pos_label):
    pred, target = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1])

    avg_prec = AveragePrecision(pos_label=pos_label)
    assert avg_prec.name == 'AP'

    ap = avg_prec(pred=pred, target=target, sample_weight=[0.1, 0.2, 0.3, 0.4])

    assert isinstance(ap, torch.Tensor)


@pytest.mark.parametrize(['pos_label'], [
    pytest.param(1.),
    pytest.param(2.)
])
def test_auroc(pos_label):
    pred, target = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1])

    auroc = AUROC(pos_label=pos_label)
    assert auroc.name == 'auroc'

    area = auroc(pred=pred, target=target, sample_weight=[0.1, 0.2, 0.3, 0.4])

    assert isinstance(area, torch.Tensor)


@pytest.mark.parametrize(['beta', 'num_classes'], [
    pytest.param(0., 1.),
    pytest.param(0.5, 1.),
    pytest.param(1., 1.),
    pytest.param(2., 1.),
    pytest.param(0., None),
    pytest.param(0.5, None),
    pytest.param(1., None),
    pytest.param(2., None)
])
def test_fbeta(beta, num_classes):
    fbeta = FBeta(beta=beta, num_classes=num_classes)
    assert fbeta.name == 'fbeta'

    score = fbeta(pred=torch.tensor([[0, 1, 1], [1, 0, 1]]),
                  target=torch.tensor([[0, 0, 1], [1, 0, 1]]))

    assert isinstance(score, torch.Tensor)


@pytest.mark.parametrize(['num_classes'], [
    pytest.param(1.),
    pytest.param(None),
])
def test_f1(num_classes):
    f1 = F1(num_classes=num_classes)
    assert f1.name == 'f1'

    score = f1(pred=torch.tensor([[0, 1, 1], [1, 0, 1]]),
               target=torch.tensor([[0, 0, 1], [1, 0, 1]]))

    assert isinstance(score, torch.Tensor)

@pytest.mark.parametrize(['pos_label'], [
    pytest.param(1.),
    pytest.param(2.)
])
def test_roc(pos_label):
    pred, target = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1])

    roc = ROC(pos_label=pos_label)
    assert roc.name == 'roc'

    r = roc(pred=pred, target=target, sample_weight=[0.1, 0.2, 0.3, 0.4])

    assert isinstance(r, tuple)
    assert len(r) == 3
    for tmp in r:
        assert isinstance(tmp, torch.Tensor)


@pytest.mark.parametrize(['num_classes'], [
    pytest.param(4.),
    pytest.param(None),

])
def test_multiclass_roc(num_classes):
    pred, target = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1])

    multi_roc = MulticlassROC(num_classes=num_classes)

    assert multi_roc.name == 'multiclass_roc'

    r = multi_roc(pred, target)

    assert isinstance(r, tuple)

    if num_classes is not None:
        assert len(r) == num_classes

    for tmp in r:
        assert isinstance(tmp, tuple)
        assert len(tmp) == 3

        for _tmp in tmp:
            assert isinstance(_tmp, torch.Tensor)


@pytest.mark.parametrize(['num_classes'], [
    pytest.param(4.),
    pytest.param(None),

])
def test_multiclass_pr(num_classes):
    pred, target = torch.tensor([1, 2, 3, 4]), torch.tensor([1, 0, 0, 1])

    multi_pr = MulticlassPrecisionRecall(num_classes=num_classes)

    assert multi_pr.name == 'multiclass_precision_recall_curve'

    pr = multi_pr(pred, target)

    assert isinstance(pr, tuple)

    if num_classes is not None:
        assert len(pr) == num_classes

    for tmp in pr:
        assert isinstance(tmp, tuple)
        assert len(tmp) == 3

        for _tmp in tmp:
            assert isinstance(_tmp, torch.Tensor)


@pytest.mark.parametrize(['include_background'],[
    pytest.param(True),
    pytest.param(False),
])
def test_dice_coefficient(include_background):
    dice_coeff = DiceCoefficient(include_background=include_background)

    assert dice_coeff.name == 'dice'

    dice = dice_coeff(torch.randint(0, 1, (10, 25, 25)),
                      torch.randint(0, 1, (10, 25, 25)))

    assert isinstance(dice, torch.Tensor)
