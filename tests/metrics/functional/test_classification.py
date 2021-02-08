import pytest
import torch

from pytorch_lightning import seed_everything
from pytorch_lightning.metrics.functional.classification import dice_score
from pytorch_lightning.metrics.functional.precision_recall_curve import _binary_clf_curve
from pytorch_lightning.metrics.utils import get_num_classes, to_categorical, to_onehot


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


@pytest.mark.parametrize(['sample_weight', 'pos_label', "exp_shape"], [
    pytest.param(1, 1., 42),
    pytest.param(None, 1., 42),
])
def test_binary_clf_curve(sample_weight, pos_label, exp_shape):
    # TODO: move back the pred and target to test func arguments
    #  if you fix the array inside the function, you'd also have fix the shape,
    #  because when the array changes, you also have to fix the shape
    seed_everything(0)
    pred = torch.randint(low=51, high=99, size=(100, ), dtype=torch.float) / 100
    target = torch.tensor([0, 1] * 50, dtype=torch.int)
    if sample_weight is not None:
        sample_weight = torch.ones_like(pred) * sample_weight

    fps, tps, thresh = _binary_clf_curve(preds=pred, target=target, sample_weights=sample_weight, pos_label=pos_label)

    assert isinstance(tps, torch.Tensor)
    assert isinstance(fps, torch.Tensor)
    assert isinstance(thresh, torch.Tensor)
    assert tps.shape == (exp_shape, )
    assert fps.shape == (exp_shape, )
    assert thresh.shape == (exp_shape, )


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([[0, 0], [1, 1]], [[0, 0], [1, 1]], 1.),
    pytest.param([[1, 1], [0, 0]], [[0, 0], [1, 1]], 0.),
    pytest.param([[1, 1], [1, 1]], [[1, 1], [0, 0]], 2 / 3),
    pytest.param([[1, 1], [0, 0]], [[1, 1], [0, 0]], 1.),
])
def test_dice_score(pred, target, expected):
    score = dice_score(torch.tensor(pred), torch.tensor(target))
    assert score == expected
