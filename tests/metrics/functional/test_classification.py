import pytest
import torch

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
)


@pytest.fixture
def random():
    torch.manual_seed(0)


def test_onehot():
    test_tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    expected = torch.tensor([
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ]
    ])

    assert test_tensor.shape == (2, 5)
    assert expected.shape == (2, 10, 5)

    onehot_classes = to_onehot(test_tensor, n_classes=10)
    onehot_no_classes = to_onehot(test_tensor)

    assert torch.allclose(onehot_classes, onehot_no_classes)

    assert onehot_classes.shape == expected.shape
    assert onehot_no_classes.shape == expected.shape

    assert torch.allclose(expected.to(onehot_no_classes), onehot_no_classes)
    assert torch.allclose(expected.to(onehot_classes), onehot_classes)


def test_to_categorical():
    test_tensor = torch.tensor([
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]
        ], [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1]
        ]
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


@pytest.mark.parametrize(['pred', 'target', 'expected_tp', 'expected_fp', 'expected_tn', 'expected_fn'], [
    pytest.param(torch.tensor([0., 2., 4., 4.]), torch.tensor([0., 4., 3., 4.]), 1, 1, 1, 1),
    pytest.param(to_onehot(torch.tensor([0., 2., 4., 4.])), torch.tensor([0., 4., 3., 4.]), 1, 1, 1, 1)
])
def test_stat_scores(pred, target, expected_tp, expected_fp, expected_tn, expected_fn):
    tp, fp, tn, fn = stat_scores(pred, target, class_index=4)

    assert tp.item() == expected_tp
    assert fp.item() == expected_fp
    assert tn.item() == expected_tn
    assert fn.item() == expected_fn


@pytest.mark.parametrize(['pred', 'target', 'expected_tp', 'expected_fp', 'expected_tn', 'expected_fn'], [
    pytest.param(torch.tensor([0., 2., 4., 4.]), torch.tensor([0., 4., 3., 4.]),
                 [1, 0, 0, 0, 1], [0, 0, 1, 0, 1], [3, 4, 3, 3, 1], [0, 0, 0, 1, 1]),
    pytest.param(to_onehot(torch.tensor([0., 2., 4., 4.])), torch.tensor([0., 4., 3., 4.]),
                 [1, 0, 0, 0, 1], [0, 0, 1, 0, 1], [3, 4, 3, 3, 1], [0, 0, 0, 1, 1])
])
def test_stat_scores_multiclass(pred, target, expected_tp, expected_fp, expected_tn, expected_fn):
    tp, fp, tn, fn = stat_scores_multiple_classes(pred, target)

    assert torch.allclose(torch.tensor(expected_tp).to(tp), tp)
    assert torch.allclose(torch.tensor(expected_fp).to(fp), fp)
    assert torch.allclose(torch.tensor(expected_tn).to(tn), tn)
    assert torch.allclose(torch.tensor(expected_fn).to(fn), fn)


def test_multilabel_accuracy():
    # Dense label indicator matrix format
    y1 = torch.tensor([[0, 1, 1], [1, 0, 1]])
    y2 = torch.tensor([[0, 0, 1], [1, 0, 1]])

    assert torch.allclose(accuracy(y1, y2, reduction='none'), torch.tensor([0.8333333134651184] * 2))
    assert torch.allclose(accuracy(y1, y1, reduction='none'), torch.tensor([1., 1.]))
    assert torch.allclose(accuracy(y2, y2, reduction='none'), torch.tensor([1., 1.]))
    assert torch.allclose(accuracy(y2, torch.logical_not(y2), reduction='none'), torch.tensor([0., 0.]))
    assert torch.allclose(accuracy(y1, torch.logical_not(y1), reduction='none'), torch.tensor([0., 0.]))

    with pytest.raises(RuntimeError):
        accuracy(y2, torch.zeros_like(y2), reduction='none')


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
    pytest.param([1., 0., 1., 0.], [0., 1., 1., 0.], [0.5, 0.5]),
])
def test_f1_score(pred, target, exp_score):
    score = f1_score(torch.tensor(pred), torch.tensor(target), reduction='none')
    assert torch.allclose(score, torch.tensor(exp_score))
    
    score = f1_score(to_onehot(torch.tensor(pred)), torch.tensor(target), reduction='none')
    assert torch.allclose(score, torch.tensor(exp_score))


@pytest.mark.parametrize(
    ['pred', 'target', 'sample_weight', 'pos_label', "exp_shape"], [
        pytest.param(torch.randint(low=51, high=99, size=(100,), dtype=torch.float) / 100,
                     torch.tensor([int(bool(idx % 2)) for idx in range(100)]),
                     torch.ones(100), 1., 40
                     ),
        pytest.param(torch.randint(low=51, high=99, size=(100,), dtype=torch.float) / 100,
                     torch.tensor([int(bool(idx % 2)) for idx in range(100)]),
                     None, 1., 39
                     ),
    ]
)
@pytest.mark.usefixtures("random")
def test_binary_clf_curve(pred, target, sample_weight, pos_label, exp_shape):
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
    pytest.param([0, 1], [0, 1], 1.),
    pytest.param([1, 0], [0, 1], 0.),
    pytest.param([1, 1], [1, 0], 0.5),
    pytest.param([1, 0], [1, 0], 1.),
    pytest.param([0.5, 0.5], [1, 0], 0.5),
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


def test_average_precision_constant_values():
    # Check the average_precision_score of a constant predictor is
    # the TPR

    # Generate a dataset with 25% of positives
    target = torch.zeros(100, dtype=torch.float)
    target[::4] = 1
    # And a constant score
    pred = torch.ones(100)
    # The precision is then the fraction of positive whatever the recall
    # is, as there is only one threshold:
    assert average_precision(pred, target).item() == .25


@pytest.mark.skip
# TODO
def test_dice_score():
    dice_score()

# example data taken from
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/tests/test_ranking.py
