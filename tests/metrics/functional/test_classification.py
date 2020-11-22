import pytest
import torch
from sklearn.metrics import (
    roc_curve as sk_roc_curve,
    roc_auc_score as sk_roc_auc_score,
    precision_recall_curve as sk_precision_recall_curve,
)

from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional.classification import (
    to_onehot,
    to_categorical,
    get_num_classes,
    _binary_clf_curve,
    average_precision,
    auroc,
    multiclass_auroc,
    precision_recall_curve,
    roc,
    auc,
)


@pytest.mark.parametrize(
    ["sklearn_metric", "torch_metric", "only_binary"],
    [
        pytest.param(sk_roc_curve, roc, True, id="roc"),
        pytest.param(sk_precision_recall_curve, precision_recall_curve, True, id="precision_recall_curve"),
        pytest.param(sk_roc_auc_score, auroc, True, id="auroc"),
    ],
)
def test_against_sklearn(sklearn_metric, torch_metric, only_binary):
    """Compare PL metrics to sklearn version. """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # for metrics with only_binary=False, we try out different combinations of number
    # of labels in pred and target (also test binary)
    # for metrics with only_binary=True, target is always binary and pred will be
    # (unnormalized) class probabilities
    class_comb = [(5, 2)] if only_binary else [(10, 10), (5, 10), (10, 5), (2, 2)]
    for n_cls_pred, n_cls_target in class_comb:
        pred = torch.randint(n_cls_pred, (300,), device=device)
        target = torch.randint(n_cls_target, (300,), device=device)

        sk_score = sklearn_metric(target.cpu().detach().numpy(), pred.cpu().detach().numpy())
        pl_score = torch_metric(pred, target)

        # if multi output
        if isinstance(sk_score, tuple):
            sk_score = [torch.tensor(sk_s.copy(), dtype=torch.float, device=device) for sk_s in sk_score]
            for sk_s, pl_s in zip(sk_score, pl_score):
                assert torch.allclose(sk_s, pl_s.float())
        else:
            sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
            assert torch.allclose(sk_score, pl_score)


def test_onehot():
    test_tensor = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    expected = torch.stack(
        [
            torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
            torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)]),
        ]
    )

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
    test_tensor = torch.stack(
        [
            torch.cat([torch.eye(5, dtype=int), torch.zeros((5, 5), dtype=int)]),
            torch.cat([torch.zeros((5, 5), dtype=int), torch.eye(5, dtype=int)]),
        ]
    ).to(torch.float)

    expected = torch.tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert expected.shape == (2, 5)
    assert test_tensor.shape == (2, 10, 5)

    result = to_categorical(test_tensor)

    assert result.shape == expected.shape
    assert torch.allclose(result, expected.to(result.dtype))


@pytest.mark.parametrize(
    ["pred", "target", "num_classes", "expected_num_classes"],
    [
        pytest.param(torch.rand(32, 10, 28, 28), torch.randint(10, (32, 28, 28)), 10, 10),
        pytest.param(torch.rand(32, 10, 28, 28), torch.randint(10, (32, 28, 28)), None, 10),
        pytest.param(torch.rand(32, 28, 28), torch.randint(10, (32, 28, 28)), None, 10),
    ],
)
def test_get_num_classes(pred, target, num_classes, expected_num_classes):
    assert get_num_classes(pred, target, num_classes) == expected_num_classes


@pytest.mark.parametrize(
    ["sample_weight", "pos_label", "exp_shape"],
    [
        pytest.param(1, 1.0, 42),
        pytest.param(None, 1.0, 42),
    ],
)
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


@pytest.mark.parametrize(
    ["pred", "target", "expected_p", "expected_r", "expected_t"],
    [pytest.param([1, 2, 3, 4], [1, 0, 0, 1], [0.5, 1 / 3, 0.5, 1.0, 1.0], [1, 0.5, 0.5, 0.5, 0.0], [1, 2, 3, 4])],
)
def test_pr_curve(pred, target, expected_p, expected_r, expected_t):
    p, r, t = precision_recall_curve(torch.tensor(pred), torch.tensor(target))
    assert p.size() == r.size()
    assert p.size(0) == t.size(0) + 1

    assert torch.allclose(p, torch.tensor(expected_p).to(p))
    assert torch.allclose(r, torch.tensor(expected_r).to(r))
    assert torch.allclose(t, torch.tensor(expected_t).to(t))


@pytest.mark.parametrize(
    ["pred", "target", "expected_tpr", "expected_fpr"],
    [
        pytest.param([0, 1], [0, 1], [0, 1, 1], [0, 0, 1]),
        pytest.param([1, 0], [0, 1], [0, 0, 1], [0, 1, 1]),
        pytest.param([1, 1], [1, 0], [0, 1], [0, 1]),
        pytest.param([1, 0], [1, 0], [0, 1, 1], [0, 0, 1]),
        pytest.param([0.5, 0.5], [0, 1], [0, 1], [0, 1]),
    ],
)
def test_roc_curve(pred, target, expected_tpr, expected_fpr):
    fpr, tpr, thresh = roc(torch.tensor(pred), torch.tensor(target))

    assert fpr.shape == tpr.shape
    assert fpr.size(0) == thresh.size(0)
    assert torch.allclose(fpr, torch.tensor(expected_fpr).to(fpr))
    assert torch.allclose(tpr, torch.tensor(expected_tpr).to(tpr))


@pytest.mark.parametrize(
    ["pred", "target", "expected"],
    [
        pytest.param([0, 1, 0, 1], [0, 1, 0, 1], 1.0),
        pytest.param([1, 1, 0, 0], [0, 0, 1, 1], 0.0),
        pytest.param([1, 1, 1, 1], [1, 1, 0, 0], 0.5),
        pytest.param([1, 1, 0, 0], [1, 1, 0, 0], 1.0),
        pytest.param([0.5, 0.5, 0.5, 0.5], [1, 1, 0, 0], 0.5),
    ],
)
def test_auroc(pred, target, expected):
    score = auroc(torch.tensor(pred), torch.tensor(target)).item()
    assert score == expected


def test_multiclass_auroc():
    with pytest.raises(ValueError, match=r".*probabilities, i.e. they should sum up to 1.0 over classes"):
        _ = multiclass_auroc(pred=torch.tensor([[0.9, 0.9], [1.0, 0]]), target=torch.tensor([0, 1]))

    with pytest.raises(ValueError, match=r".*not defined when all of the classes do not occur in the target.*"):
        _ = multiclass_auroc(pred=torch.rand((4, 3)).softmax(dim=1), target=torch.tensor([1, 0, 1, 0]))

    with pytest.raises(ValueError, match=r".*does not equal the number of classes passed in 'num_classes'.*"):
        _ = multiclass_auroc(
            pred=torch.rand((5, 4)).softmax(dim=1), target=torch.tensor([0, 1, 2, 2, 3]), num_classes=6
        )


@pytest.mark.parametrize("n_cls", [2, 5, 10, 50])
def test_multiclass_auroc_against_sklearn(n_cls):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    n_samples = 300
    pred = torch.rand(n_samples, n_cls, device=device).softmax(dim=1)
    target = torch.randint(n_cls, (n_samples,), device=device)
    # Make sure target includes all class labels so that multiclass AUROC is defined
    target[10 : 10 + n_cls] = torch.arange(n_cls)

    pl_score = multiclass_auroc(pred, target)
    # For the binary case, sklearn expects an (n_samples,) array of probabilities of
    # the positive class
    pred = pred[:, 1] if n_cls == 2 else pred
    sk_score = sk_roc_auc_score(target.cpu().detach().numpy(), pred.cpu().detach().numpy(), multi_class="ovr")

    sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
    assert torch.allclose(sk_score, pl_score)


@pytest.mark.parametrize(
    ["x", "y", "expected"],
    [
        pytest.param([0, 1], [0, 1], 0.5),
        pytest.param([1, 0], [0, 1], 0.5),
        pytest.param([1, 0, 0], [0, 1, 1], 0.5),
        pytest.param([0, 1], [1, 1], 1),
        pytest.param([0, 0.5, 1], [0, 0.5, 1], 0.5),
    ],
)
def test_auc(x, y, expected):
    # Test Area Under Curve (AUC) computation
    assert auc(torch.tensor(x), torch.tensor(y)) == expected


@pytest.mark.parametrize(
    ["scores", "target", "expected_score"],
    [
        # Check the average_precision_score of a constant predictor is
        # the TPR
        # Generate a dataset with 25% of positives
        # And a constant score
        # The precision is then the fraction of positive whatever the recall
        # is, as there is only one threshold:
        pytest.param(torch.tensor([1, 1, 1, 1]), torch.tensor([0, 0, 0, 1]), 0.25),
        # With threshold 0.8 : 1 TP and 2 TN and one FN
        pytest.param(torch.tensor([0.6, 0.7, 0.8, 9]), torch.tensor([1, 0, 0, 1]), 0.75),
    ],
)
def test_average_precision(scores, target, expected_score):
    assert average_precision(scores, target) == expected_score


@pytest.mark.parametrize("metric", [auroc])
def test_error_on_multiclass_input(metric):
    """ check that these metrics raise an error if they are used for multiclass problems  """
    pred = torch.randint(0, 10, (100,))
    target = torch.randint(0, 10, (100,))
    with pytest.raises(ValueError, match="AUROC metric is meant for binary classification"):
        _ = metric(pred, target)
