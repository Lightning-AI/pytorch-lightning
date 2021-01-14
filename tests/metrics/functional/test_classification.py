from functools import partial

import pytest
import torch
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score as sk_recall

from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional.classification import (
    dice_score,
    precision,
    recall,
    stat_scores,
    stat_scores_multiple_classes,
)
from pytorch_lightning.metrics.functional.precision_recall_curve import _binary_clf_curve
from pytorch_lightning.metrics.utils import get_num_classes, to_categorical, to_onehot


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric', 'only_binary'], [
    pytest.param(partial(sk_precision, average='micro'), precision, False, id='precision'),
    pytest.param(partial(sk_recall, average='micro'), recall, False, id='recall'),
])
def test_against_sklearn(sklearn_metric, torch_metric, only_binary):
    """Compare PL metrics to sklearn version. """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # for metrics with only_binary=False, we try out different combinations of number
    # of labels in pred and target (also test binary)
    # for metrics with only_binary=True, target is always binary and pred will be
    # (unnormalized) class probabilities
    class_comb = [(5, 2)] if only_binary else [(10, 10), (5, 10), (10, 5), (2, 2)]
    for n_cls_pred, n_cls_target in class_comb:
        pred = torch.randint(n_cls_pred, (300,), device=device)
        target = torch.randint(n_cls_target, (300,), device=device)

        sk_score = sklearn_metric(target.cpu().detach().numpy(),
                                  pred.cpu().detach().numpy())
        pl_score = torch_metric(pred, target)

        # if multi output
        if isinstance(sk_score, tuple):
            sk_score = [torch.tensor(sk_s.copy(), dtype=torch.float, device=device) for sk_s in sk_score]
            for sk_s, pl_s in zip(sk_score, pl_score):
                assert torch.allclose(sk_s, pl_s.float())
        else:
            sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
            assert torch.allclose(sk_score, pl_score)


@pytest.mark.parametrize('class_reduction', ['micro', 'macro', 'weighted'])
@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    pytest.param(sk_precision, precision, id='precision'),
    pytest.param(sk_recall, recall, id='recall'),
])
def test_different_reduction_against_sklearn(class_reduction, sklearn_metric, torch_metric):
    """ Test metrics where the class_reduction parameter have a correponding
        value in sklearn """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred = torch.randint(10, (300,), device=device)
    target = torch.randint(10, (300,), device=device)
    sk_score = sklearn_metric(target.cpu().detach().numpy(),
                              pred.cpu().detach().numpy(),
                              average=class_reduction)
    sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
    pl_score = torch_metric(pred, target, class_reduction=class_reduction)
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
def test_stat_scores_multiclass(pred, target, reduction,
                                expected_tp, expected_fp, expected_tn, expected_fn, expected_support):
    tp, fp, tn, fn, sup = stat_scores_multiple_classes(pred, target, reduction=reduction)

    assert torch.allclose(torch.tensor(expected_tp).to(tp), tp)
    assert torch.allclose(torch.tensor(expected_fp).to(fp), fp)
    assert torch.allclose(torch.tensor(expected_tn).to(tn), tn)
    assert torch.allclose(torch.tensor(expected_fn).to(fn), fn)
    assert torch.allclose(torch.tensor(expected_support).to(sup), sup)


@pytest.mark.parametrize(['pred', 'target', 'expected_prec', 'expected_rec'], [
    pytest.param(torch.tensor([1., 0., 1., 0.]), torch.tensor([0., 1., 1., 0.]), [0.5, 0.5], [0.5, 0.5]),
    pytest.param(to_onehot(torch.tensor([1., 0., 1., 0.])), torch.tensor([0., 1., 1., 0.]), [0.5, 0.5], [0.5, 0.5])
])
def test_precision_recall(pred, target, expected_prec, expected_rec):
    prec = precision(pred, target, class_reduction='none')
    rec = recall(pred, target, class_reduction='none')

    assert torch.allclose(torch.tensor(expected_prec).to(prec), prec)
    assert torch.allclose(torch.tensor(expected_rec).to(rec), rec)


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

    fps, tps, thresh = _binary_clf_curve(preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label)

    assert isinstance(tps, torch.Tensor)
    assert isinstance(fps, torch.Tensor)
    assert isinstance(thresh, torch.Tensor)
    assert tps.shape == (exp_shape,)
    assert fps.shape == (exp_shape,)
    assert thresh.shape == (exp_shape,)


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1.),
    pytest.param([[1, 1], [0, 0]], [[0, 0], [1, 1]], 0.),
    pytest.param([[1, 1], [1, 1]], [[1, 1], [0, 0]], 2 / 3),
    pytest.param([[1, 1], [0, 0]], [[1, 1], [0, 0]], 1.),
])
def test_dice_score(pred, target, expected):
    score = dice_score(torch.tensor(pred), torch.tensor(target))
    assert score == expected
