import sys
from pytorch_lightning.metrics.functional.iou import iou
import pytest
import torch
from functools import partial
from sklearn.metrics import jaccard_score as sk_jaccard_score

# TODO: combine this function with the one in test_classification.py
@pytest.mark.parametrize(['sklearn_metric', 'torch_metric', 'only_binary'], [
    pytest.param(partial(sk_jaccard_score, average='macro'), iou, False, id='iou'),
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


@pytest.mark.parametrize(['half_ones', 'reduction', 'ignore_index', 'expected'], [
    pytest.param(False, 'none', None, torch.Tensor([1, 1, 1])),
    pytest.param(False, 'elementwise_mean', None, torch.Tensor([1])),
    pytest.param(False, 'none', 0, torch.Tensor([1, 1])),
    pytest.param(True, 'none', None, torch.Tensor([0.5, 0.5, 0.5])),
    pytest.param(True, 'elementwise_mean', None, torch.Tensor([0.5])),
    pytest.param(True, 'none', 0, torch.Tensor([0.5, 0.5])),
])
def test_iou(half_ones, reduction, ignore_index, expected):
    pred = (torch.arange(120) % 3).view(-1, 1)
    target = (torch.arange(120) % 3).view(-1, 1)
    if half_ones:
        pred[:60] = 1
    iou_val = iou(
        pred=pred,
        target=target,
        ignore_index=ignore_index,
        reduction=reduction,
    )
    assert torch.allclose(iou_val, expected, atol=1e-9)


def test_iou_input_check():
    with pytest.raises(ValueError, match=r"'pred' shape (.*) must equal 'target' shape (.*)"):
        _ = iou(pred=torch.randint(0, 2, (3, 4, 3)),
                target=torch.randint(0, 2, (3, 3)))

    with pytest.raises(ValueError, match="'pred' must contain integer targets."):
        _ = iou(pred=torch.rand((3, 3)),
                target=torch.randint(0, 2, (3, 3)))


# TODO: When the jaccard_score of the sklearn version we use accepts `zero_division` (see
#       https://github.com/scikit-learn/scikit-learn/pull/17866), consider adding a test here against our
#       `absent_score`.
@pytest.mark.parametrize(['pred', 'target', 'ignore_index', 'absent_score', 'num_classes', 'expected'], [
    # Note that -1 is used as the absent_score in almost all tests here to distinguish it from the range of valid
    # scores the function can return ([0., 1.] range, inclusive).
    # 2 classes, class 0 is correct everywhere, class 1 is absent.
    pytest.param([0], [0], None, -1., 2, [1., -1.]),
    pytest.param([0, 0], [0, 0], None, -1., 2, [1., -1.]),
    # absent_score not applied if only class 0 is present and it's the only class.
    pytest.param([0], [0], None, -1., 1, [1.]),
    # 2 classes, class 1 is correct everywhere, class 0 is absent.
    pytest.param([1], [1], None, -1., 2, [-1., 1.]),
    pytest.param([1, 1], [1, 1], None, -1., 2, [-1., 1.]),
    # When 0 index ignored, class 0 does not get a score (not even the absent_score).
    pytest.param([1], [1], 0, -1., 2, [1.0]),
    # 3 classes. Only 0 and 2 are present, and are perfectly predicted. 1 should get absent_score.
    pytest.param([0, 2], [0, 2], None, -1., 3, [1., -1., 1.]),
    pytest.param([2, 0], [2, 0], None, -1., 3, [1., -1., 1.]),
    # 3 classes. Only 0 and 1 are present, and are perfectly predicted. 2 should get absent_score.
    pytest.param([0, 1], [0, 1], None, -1., 3, [1., 1., -1.]),
    pytest.param([1, 0], [1, 0], None, -1., 3, [1., 1., -1.]),
    # 3 classes, class 0 is 0.5 IoU, class 1 is 0 IoU (in pred but not target; should not get absent_score), class
    # 2 is absent.
    pytest.param([0, 1], [0, 0], None, -1., 3, [0.5, 0., -1.]),
    # 3 classes, class 0 is 0.5 IoU, class 1 is 0 IoU (in target but not pred; should not get absent_score), class
    # 2 is absent.
    pytest.param([0, 0], [0, 1], None, -1., 3, [0.5, 0., -1.]),
    # Sanity checks with absent_score of 1.0.
    pytest.param([0, 2], [0, 2], None, 1.0, 3, [1., 1., 1.]),
    pytest.param([0, 2], [0, 2], 0, 1.0, 3, [1., 1.]),
])
def test_iou_absent_score(pred, target, ignore_index, absent_score, num_classes, expected):
    iou_val = iou(
        pred=torch.tensor(pred),
        target=torch.tensor(target),
        ignore_index=ignore_index,
        absent_score=absent_score,
        num_classes=num_classes,
        reduction='none',
    )
    assert torch.allclose(iou_val, torch.tensor(expected).to(iou_val))


# example data taken from
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/metrics/tests/test_ranking.py
@pytest.mark.parametrize(['pred', 'target', 'ignore_index', 'num_classes', 'reduction', 'expected'], [
    # Ignoring an index outside of [0, num_classes-1] should have no effect.
    pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], None, 3, 'none', [1, 1 / 2, 2 / 3]),
    pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], -1, 3, 'none', [1, 1 / 2, 2 / 3]),
    pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 255, 3, 'none', [1, 1 / 2, 2 / 3]),
    # Ignoring a valid index drops only that index from the result.
    pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, 'none', [1 / 2, 2 / 3]),
    pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 1, 3, 'none', [1, 2 / 3]),
    pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 2, 3, 'none', [1, 1 / 2]),
    # When reducing to mean or sum, the ignored index does not contribute to the output.
    pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, 'elementwise_mean', [7 / 12]),
    pytest.param([0, 1, 1, 2, 2], [0, 1, 2, 2, 2], 0, 3, 'sum', [7 / 6]),
])
def test_iou_ignore_index(pred, target, ignore_index, num_classes, reduction, expected):
    iou_val = iou(
        pred=torch.tensor(pred),
        target=torch.tensor(target),
        ignore_index=ignore_index,
        num_classes=num_classes,
        reduction=reduction,
    )
    assert torch.allclose(iou_val, torch.tensor(expected).to(iou_val))
