import pytest
import torch

from pytorch_lightning.metrics.functional.classification import to_onehot, to_categorical, get_num_classes, \
    stat_scores, stat_scores_multiple_classes, accuracy, confusion_matrix, precision, recall, fbeta_score


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
                 torch.tensor([1, 0, 0, 0, 1]), torch.tensor([0, 0, 1, 0, 1]), torch.tensor([3, 4, 3, 3, 1]),
                 torch.tensor([0, 0, 0, 1, 1])),
    pytest.param(to_onehot(torch.tensor([0., 2., 4., 4.])), torch.tensor([0., 4., 3., 4.]),
                 torch.tensor([1, 0, 0, 0, 1]), torch.tensor([0, 0, 1, 0, 1]), torch.tensor([3, 4, 3, 3, 1]),
                 torch.tensor([0, 0, 0, 1, 1]))
])
def test_stat_scores_multiclass(pred, target, expected_tp, expected_fp, expected_tn, expected_fn):
    tp, fp, tn, fn = stat_scores_multiple_classes(pred, target)

    assert torch.allclose(expected_tp.to(tp), tp)
    assert torch.allclose(expected_fp.to(fp), fp)
    assert torch.allclose(expected_tn.to(tn), tn)
    assert torch.allclose(expected_fn.to(fn), fn)


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
    pytest.param(torch.tensor([1., 0., 1., 0.]), torch.tensor([0., 1., 1., 0.]),
                 torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5])),
    pytest.param(to_onehot(torch.tensor([1., 0., 1., 0.])), torch.tensor([0., 1., 1., 0.]),
                 torch.tensor([0.5, 0.5]), torch.tensor([0.5, 0.5]))
])
def test_precision_recall(pred, target, expected_prec, expected_rec):
    prec = precision(pred, target, reduction='none')
    rec = recall(pred, target, reduction='none')

    assert torch.allclose(expected_prec.to(prec), prec)
    assert torch.allclose(expected_rec.to(rec), rec)


@pytest.mark.parametrize(['pred', 'target', 'beta', 'exp_score'], [
    pytest.param(torch.tensor([1., 0., 1., 0.]), torch.tensor([0., 1., 1., 0.]),
                 0.5, torch.tensor([0.5, 0.5])),
    pytest.param(to_onehot(torch.tensor([1., 0., 1., 0.])), torch.tensor([0., 1., 1., 0.]),
                 0.5, torch.tensor([0.5, 0.5])),
    pytest.param(torch.tensor([1., 0., 1., 0.]), torch.tensor([0., 1., 1., 0.]),
                 1, torch.tensor([0.5, 0.5])),
    pytest.param(to_onehot(torch.tensor([1., 0., 1., 0.])), torch.tensor([0., 1., 1., 0.]),
                 1, torch.tensor([0.5, 0.5])),
    pytest.param(torch.tensor([1., 0., 1., 0.]), torch.tensor([0., 1., 1., 0.]),
                 2, torch.tensor([0.5, 0.5])),
    pytest.param(to_onehot(torch.tensor([1., 0., 1., 0.])), torch.tensor([0., 1., 1., 0.]),
                 2, torch.tensor([0.5, 0.5])),
])
def test_fbeta_score(pred, target, beta, exp_score):
    score = fbeta_score(pred, target, beta, reduction='none')
    assert torch.allclose(score, exp_score)
