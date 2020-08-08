from functools import partial

import pytest
import torch
from sklearn.metrics import (
    accuracy_score as sk_accuracy,
    precision_score as sk_precision,
    recall_score as sk_recall,
    f1_score as sk_f1_score,
    fbeta_score as sk_fbeta_score,
    confusion_matrix as sk_confusion_matrix,
)

from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional.classification import (
    to_onehot,
    to_categorical,
    get_num_classes,
    stat_scores,
    stat_scores_multiple_classes,
    accuracy,
    confusion_matrix,
    precision,
    recall,
    fbeta_score,
    f1_score,
    _binary_clf_curve,
    dice_score,
    average_precision,
    auroc,
    precision_recall_curve,
    roc,
    auc,
    iou,
)


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    pytest.param(sk_accuracy, accuracy, id='accuracy'),
    pytest.param(partial(sk_precision, average='macro'), precision, id='precision'),
    pytest.param(partial(sk_recall, average='macro'), recall, id='recall'),
    pytest.param(partial(sk_f1_score, average='macro'), f1_score, id='f1_score'),
    pytest.param(partial(sk_fbeta_score, average='macro', beta=2), partial(fbeta_score, beta=2), id='fbeta_score'),
    pytest.param(sk_confusion_matrix, confusion_matrix, id='confusion_matrix')
])
def test_against_sklearn(sklearn_metric, torch_metric):
    """Compare PL metrics to sklearn version."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # iterate over different label counts in predictions and target
    for n_cls_pred, n_cls_target in [(10, 10), (5, 10), (10, 5)]:
        pred = torch.randint(n_cls_pred, (300,), device=device)
        target = torch.randint(n_cls_target, (300,), device=device)

        sk_score = sklearn_metric(target.cpu().detach().numpy(),
                                  pred.cpu().detach().numpy())
        sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
        pl_score = torch_metric(pred, target)
        assert torch.allclose(sk_score, pl_score)


def test_onehot():
    test_tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    expected = torch.stack([
        torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
        torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)])
    ])

    assert test_tensor.shape == (2, 5)
    assert expected.shape == (2, 10, 5)

    onehot_classes = to_onehot(test_tensor, num_classes=10)
    onehot_no_classes = to_onehot(test_tensor)

    assert torch.allclose(onehot_classes, onehot_no_classes)

    assert onehot_classes.shape == expected.shape
    assert onehot_no_classes.shape == expected.shape

    assert torch.allclose(expected.to(onehot_no_classes), onehot_no_classes)
    assert torch.allclose(expected.to(onehot_classes), onehot_classes)


def test_to_categorical():
    test_tensor = torch.stack([
        torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
        torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)])
    ]).to(torch.float)

    expected = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert expected.shape == (2, 5)
    assert test_tensor.shape == (2, 10, 5)

    result = to_categorical(test_tensor)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected.to(result.dtype))


@pytest.mark.parametrize(['pred', 'target', 'num_classes', 'expected_num_classes'], [
    pytest.param(torch.rand(32, 10, 28, 28), torch.randint(10, (32, 28, 28)), 10, 10),
    pytest.param(torch.rand(32, 10, 28, 28), torch.randint(10, (32, 28, 28)), None, 10),
    pytest.param(torch.rand(32, 28, 28), torch.randint(10, (32, 28, 28)), None, 10),
])
def test_get_num_classes(pred, target, num_classes, expected_num_classes):
    assert get_num_classes(pred, target, num_classes) == expected_num_classes


@pytest.mark.parametrize(['pred', 'target', 'expected_tp', 'expected_fp',
                          'expected_tn', 'expected_fn', 'expected_support'], [
    pytest.param(torch.tensor([0., 2., 4., 4.]), torch.tensor([0., 4., 3., 4.]), 1, 1, 1, 1, 2),
    pytest.param(to_onehot(torch.tensor([0., 2., 4., 4.])), torch.tensor([0., 4., 3., 4.]), 1, 1, 1, 1, 2)
])
def test_stat_scores(pred, target, expected_tp, expected_fp, expected_tn, expected_fn, expected_support):
    tp, fp, tn, fn, sup = stat_scores(pred, target, class_index=4)

    assert tp.item() == expected_tp
    assert fp.item() == expected_fp
    assert tn.item() == expected_tn
    assert fn.item() == expected_fn
    assert sup.item() == expected_support


@pytest.mark.parametrize(['pred', 'target', 'reduction', 'expected_tp', 'expected_fp',
                          'expected_tn', 'expected_fn', 'expected_support'], [
    pytest.param(torch.tensor([0., 2., 4., 4.]), torch.tensor([0., 4., 3., 4.]), 'none',
                 [1, 0, 0, 0, 1], [0, 0, 1, 0, 1], [3, 4, 3, 3, 1], [0, 0, 0, 1, 1], [1, 0, 0, 1, 2]),
    pytest.param(to_onehot(torch.tensor([0., 2., 4., 4.])), torch.tensor([0., 4., 3., 4.]), 'none',
                 [1, 0, 0, 0, 1], [0, 0, 1, 0, 1], [3, 4, 3, 3, 1], [0, 0, 0, 1, 1], [1, 0, 0, 1, 2]),
    pytest.param(to_onehot(torch.tensor([0., 2., 4., 4.])), torch.tensor([0., 4., 3., 4.]), 'sum',
                 torch.tensor(2), torch.tensor(2), torch.tensor(14), torch.tensor(2), torch.tensor(4)),
    pytest.param(to_onehot(torch.tensor([0., 2., 4., 4.])), torch.tensor([0., 4., 3., 4.]), 'elementwise_mean',
                 torch.tensor(0.4), torch.tensor(0.4), torch.tensor(2.8), torch.tensor(0.4), torch.tensor(0.8))
])
def test_stat_scores_multiclass(pred, target, reduction, expected_tp, expected_fp, expected_tn, expected_fn, expected_support):
    tp, fp, tn, fn, sup = stat_scores_multiple_classes(pred, target, reduction=reduction)

    assert torch.allclose(torch.tensor(expected_tp).to(tp), tp)
    assert torch.allclose(torch.tensor(expected_fp).to(fp), fp)
    assert torch.allclose(torch.tensor(expected_tn).to(tn), tn)
    assert torch.allclose(torch.tensor(expected_fn).to(fn), fn)
    assert torch.allclose(torch.tensor(expected_support).to(sup), sup)


def test_multilabel_accuracy():
    # Dense label indicator matrix format
    y1 = torch.tensor([[0, 1, 1], [1, 0, 1]])
    y2 = torch.tensor([[0, 0, 1], [1, 0, 1]])

    assert torch.allclose(accuracy(y1, y2, reduction='none'), torch.tensor([2 / 3, 1.]))
    assert torch.allclose(accuracy(y1, y1, reduction='none'), torch.tensor([1., 1.]))
    assert torch.allclose(accuracy(y2, y2, reduction='none'), torch.tensor([1., 1.]))
    assert torch.allclose(accuracy(y2, torch.logical_not(y2), reduction='none'), torch.tensor([0., 0.]))
    assert torch.allclose(accuracy(y1, torch.logical_not(y1), reduction='none'), torch.tensor([0., 0.]))

    with pytest.raises(RuntimeError):
        accuracy(y2, torch.zeros_like(y2), reduction='none')


def test_accuracy():
    pred = torch.tensor([0, 1, 2, 3])
    target = torch.tensor([0, 1, 2, 2])
    acc = accuracy(pred, target)

    assert acc.item() == 0.75

    pred = torch.tensor([0, 1, 2, 2])
    target = torch.tensor([0, 1, 1, 3])
    acc = accuracy(pred, target)

    assert acc.item() == 0.50


def test_confusion_matrix():
    target = (torch.arange(120) % 3).view(-1, 1)
    pred = target.clone()
    cm = confusion_matrix(pred, target, normalize=True)

    assert torch.allclose(cm, torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]))

    pred = torch.zeros_like(pred)
    cm = confusion_matrix(pred, target, normalize=True)
    assert torch.allclose(cm, torch.tensor([[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]]))


@pytest.mark.parametrize(['pred', 'target', 'expected_prec', 'expected_rec'], [
    pytest.param(torch.tensor([1., 0., 1., 0.]), torch.tensor([0., 1., 1., 0.]), [0.5, 0.5], [0.5, 0.5]),
    pytest.param(to_onehot(torch.tensor([1., 0., 1., 0.])), torch.tensor([0., 1., 1., 0.]), [0.5, 0.5], [0.5, 0.5])
])
def test_precision_recall(pred, target, expected_prec, expected_rec):
    prec = precision(pred, target, reduction='none')
    rec = recall(pred, target, reduction='none')

    assert torch.allclose(torch.tensor(expected_prec).to(prec), prec)
    assert torch.allclose(torch.tensor(expected_rec).to(rec), rec)


@pytest.mark.parametrize(['pred', 'target', 'beta', 'exp_score'], [
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], 0.5, [0.5, 0.5]),
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], 1, [0.5, 0.5]),
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], 2, [0.5, 0.5]),
])
def test_fbeta_score(pred, target, beta, exp_score):
    score = fbeta_score(torch.tensor(pred), torch.tensor(target), beta, reduction='none')
    assert torch.allclose(score, torch.tensor(exp_score))

    score = fbeta_score(to_onehot(torch.tensor(pred)), torch.tensor(target), beta, reduction='none')
    assert torch.allclose(score, torch.tensor(exp_score))


@pytest.mark.parametrize(['pred', 'target', 'exp_score'], [
    pytest.param([0., 0., 0., 0.], [1., 1., 1., 1.], [0.0, 0.0]),
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], [0.5, 0.5]),
    pytest.param([1., 0., 1., 0.], [1., 0., 1., 0.], [1.0, 1.0]),
])
def test_f1_score(pred, target, exp_score):
    score = f1_score(torch.tensor(pred), torch.tensor(target), reduction='none')
    assert torch.allclose(score, torch.tensor(exp_score))

    score = f1_score(to_onehot(torch.tensor(pred)), torch.tensor(target), reduction='none')
    assert torch.allclose(score, torch.tensor(exp_score))


@pytest.mark.parametrize(['sample_weight', 'pos_label', "exp_shape"], [
    pytest.param(1, 1., 42),
    pytest.param(None, 1., 42),
])
def test_binary_clf_curve(sample_weight, pos_label, exp_shape):
    # TODO: move back the pred and target to test func arguments
    #  if you fix the array inside the function, you'd also have fix the shape,
    #  because when the array changes, you also have to fix the shape
    seed_everything(0)
    pred = torch.randint(low=51, high=99, size=(100,), dtype=torch.float) / 100
    target = torch.tensor([0, 1] * 50, dtype=torch.int)
    if sample_weight is not None:
        sample_weight = torch.ones_like(pred) * sample_weight

    fps, tps, thresh = _binary_clf_curve(pred, target, sample_weight, pos_label)

    assert isinstance(tps, torch.Tensor)
    assert isinstance(fps, torch.Tensor)
    assert isinstance(thresh, torch.Tensor)
    assert tps.shape == (exp_shape,)
    assert fps.shape == (exp_shape,)
    assert thresh.shape == (exp_shape,)


@pytest.mark.parametrize(['pred', 'target', 'expected_p', 'expected_r', 'expected_t'], [
    pytest.param([1, 2, 3, 4], [1, 0, 0, 1], [0.5, 1 / 3, 0.5, 1., 1.], [1, 0.5, 0.5, 0.5, 0.], [1, 2, 3, 4])
])
def test_pr_curve(pred, target, expected_p, expected_r, expected_t):
    p, r, t = precision_recall_curve(torch.tensor(pred), torch.tensor(target))
    assert p.size() == r.size()
    assert p.size(0) == t.size(0) + 1

    assert torch.allclose(p, torch.tensor(expected_p).to(p))
    assert torch.allclose(r, torch.tensor(expected_r).to(r))
    assert torch.allclose(t, torch.tensor(expected_t).to(t))


@pytest.mark.parametrize(['pred', 'target', 'expected_tpr', 'expected_fpr'], [
    pytest.param([0, 1], [0, 1], [0, 1, 1], [0, 0, 1]),
    pytest.param([1, 0], [0, 1], [0, 0, 1], [0, 1, 1]),
    pytest.param([1, 1], [1, 0], [0, 1], [0, 1]),
    pytest.param([1, 0], [1, 0], [0, 1, 1], [0, 0, 1]),
    pytest.param([0.5, 0.5], [0, 1], [0, 1], [0, 1]),
])
def test_roc_curve(pred, target, expected_tpr, expected_fpr):
    fpr, tpr, thresh = roc(torch.tensor(pred), torch.tensor(target))

    assert fpr.shape == tpr.shape
    assert fpr.size(0) == thresh.size(0)
    assert torch.allclose(fpr, torch.tensor(expected_fpr).to(fpr))
    assert torch.allclose(tpr, torch.tensor(expected_tpr).to(tpr))


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([0, 1, 0, 1], [0, 1, 0, 1], 1.),
    pytest.param([1, 1, 0, 0], [0, 0, 1, 1], 0.),
    pytest.param([1, 1, 1, 1], [1, 1, 0, 0], 0.5),
    pytest.param([1, 1, 0, 0], [1, 1, 0, 0], 1.),
    pytest.param([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0], 0.5),
])
def test_auroc(pred, target, expected):
    score = auroc(torch.tensor(pred), torch.tensor(target)).item()
    assert score == expected


@pytest.mark.parametrize(['x', 'y', 'expected'], [
    pytest.param([0, 1], [0, 1], 0.5),
    pytest.param([1, 0], [0, 1], 0.5),
    pytest.param([1, 0, 0], [0, 1, 1], 0.5),
    pytest.param([0, 1], [1, 1], 1),
    pytest.param([0, 0.5, 1], [0, 0.5, 1], 0.5),
])
def test_auc(x, y, expected):
    # Test Area Under Curve (AUC) computation
    assert auc(torch.tensor(x), torch.tensor(y)) == expected


@pytest.mark.parametrize(['scores', 'target', 'expected_score'], [
    # Check the average_precision_score of a constant predictor is
    # the TPR
    # Generate a dataset with 25% of positives
    # And a constant score
    # The precision is then the fraction of positive whatever the recall
    # is, as there is only one threshold:
    pytest.param(torch.tensor([1, 1, 1, 1]), torch.tensor([0, 0, 0, 1]), .25),
    # With threshold 0.8 : 1 TP and 2 TN and one FN
    pytest.param(torch.tensor([.6, .7, .8, 9]), torch.tensor([1, 0, 0, 1]), .75),
])
def test_average_precision(scores, target, expected_score):
    assert average_precision(scores, target) == expected_score


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1.),
    pytest.param([[1, 1], [0, 0]], [[0, 0], [1, 1]], 0.),
    pytest.param([[1, 1], [1, 1]], [[1, 1], [0, 0]], 2 / 3),
    pytest.param([[1, 1], [0, 0]], [[1, 1], [0, 0]], 1.),
])
def test_dice_score(pred, target, expected):
    score = dice_score(torch.tensor(pred), torch.tensor(target))
    assert score == expected


@pytest.mark.parametrize(['half_ones', 'reduction', 'remove_bg', 'expected'], [
    pytest.param(False, 'none', False, torch.Tensor([1, 1, 1])),
    pytest.param(False, 'elementwise_mean', False, torch.Tensor([1])),
    pytest.param(False, 'none', True, torch.Tensor([1, 1])),
    pytest.param(True, 'none', False, torch.Tensor([0.5, 0.5, 0.5])),
    pytest.param(True, 'elementwise_mean', False, torch.Tensor([0.5])),
    pytest.param(True, 'none', True, torch.Tensor([0.5, 0.5])),
])
def test_iou(half_ones, reduction, remove_bg, expected):
    pred = (torch.arange(120) % 3).view(-1, 1)
    target = (torch.arange(120) % 3).view(-1, 1)
    if half_ones:
        pred[:60] = 1
    iou_val = iou(pred, target, remove_bg=remove_bg, reduction=reduction)
    assert torch.allclose(iou_val, expected, atol=1e-9)


# example data taken from
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/tests/test_ranking.py
