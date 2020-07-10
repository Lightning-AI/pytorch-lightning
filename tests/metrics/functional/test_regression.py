import pytest
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

from pytorch_lightning.metrics.functional import (
    mae,
    mse,
    psnr,
    rmse,
    rmsle
)


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([0., 1, 2, 3], [0., 1, 2, 2], 0.25),
    pytest.param([4., 3, 2, 1], [1., 4, 3, 2], 3.0),
])
def test_mse(pred, target, expected):
    score = mse(torch.tensor(pred), torch.tensor(target))
    assert score.item() == expected


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([0., 1, 2, 3], [0., 1, 2, 3], 0.0),
    pytest.param([0., 1, 2, 3], [0., 1, 2, 2], 0.5),
    pytest.param([4., 3, 2, 1], [1., 4, 3, 2], 1.7321),
])
def test_rmse(pred, target, expected):
    score = rmse(torch.tensor(pred), torch.tensor(target))
    assert torch.allclose(score, torch.tensor(expected), atol=1e-3)


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([0., 1, 2, 3], [0., 1, 2, 3], 0.0),
    pytest.param([0., 1, 2, 3], [0., 1, 2, 2], 0.25),
    pytest.param([4., 3, 2, 1], [1., 4, 3, 2], 1.5),
])
def test_mae(pred, target, expected):
    score = mae(torch.tensor(pred), torch.tensor(target))
    assert score.item() == expected


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([0., 1, 2, 3], [0., 1, 2, 3], 0.0),
    pytest.param([0., 1, 2, 3], [0., 1, 2, 2], 0.0207),
    pytest.param([4., 3, 2, 1], [1., 4, 3, 2], 0.2841),
])
def test_rmsle(pred, target, expected):
    score = rmsle(torch.tensor(pred), torch.tensor(target))
    assert torch.allclose(score, torch.tensor(expected), atol=1e-3)


@pytest.mark.parametrize(['pred', 'target'], [
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 3.]),
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 2.]),
    pytest.param([4., 3., 2., 1.], [1., 4., 3., 2.]),
])
def test_psnr_with_skimage(pred, target):
    score = psnr(pred=torch.tensor(pred),
                 target=torch.tensor(target))
    sk_score = ski_psnr(np.array(pred), np.array(target), data_range=3)
    assert torch.allclose(score, torch.tensor(sk_score, dtype=torch.float), atol=1e-3)


@pytest.mark.parametrize(['pred', 'target'], [
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 2.]),
    pytest.param([4., 3., 2., 1.], [1., 4., 3., 2.]),
])
def test_psnr_base_e_wider_range(pred, target):
    score = psnr(pred=torch.tensor(pred),
                 target=torch.tensor(target),
                 data_range=4,
                 base=2.718281828459045)
    sk_score = ski_psnr(np.array(pred), np.array(target), data_range=4) * np.log(10)
    assert torch.allclose(score, torch.tensor(sk_score, dtype=torch.float32), atol=1e-3)


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    pytest.param(ski_psnr, psnr, id='peak_signal_noise_ratio')
])
def test_psnr_against_sklearn(sklearn_metric, torch_metric):
    """Compare PL metrics to sklearn version."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for n_cls_pred, n_cls_target in [(10, 10), (5, 10), (10, 5)]:
        pred = torch.randint(n_cls_pred, (500,), device=device, dtype=torch.float)
        target = torch.randint(n_cls_target, (500,), device=device, dtype=torch.float)
    
        sk_score = sklearn_metric(target.cpu().detach().numpy(),
                                  pred.cpu().detach().numpy(),
                                  data_range=n_cls_target)
        sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
        pl_score = torch_metric(pred, target, data_range=n_cls_target)
        assert torch.allclose(sk_score, pl_score)
