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
    pytest.param([4., 3, 2, 1], [1., 4, 3, 2], 3.0)
])
def test_mse(pred, target, expected):
    score = mse(torch.tensor(pred), torch.tensor(target))
    assert score.item() == expected


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([0., 1, 2, 3], [0., 1, 2, 2], 0.5),
    pytest.param([4., 3, 2, 1], [1., 4, 3, 2], 1.7321)
])
def test_rmse(pred, target, expected):
    score = rmse(torch.tensor(pred), torch.tensor(target))
    assert torch.allclose(score, torch.tensor(expected), atol=1e-3)


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([0., 1, 2, 3], [0., 1, 2, 2], 0.25),
    pytest.param([4., 3, 2, 1], [1., 4, 3, 2], 1.5)
])
def test_mae(pred, target, expected):
    score = mae(torch.tensor(pred), torch.tensor(target))
    assert score.item() == expected


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param([0., 1, 2, 3], [0., 1, 2, 2], 0.0207),
    pytest.param([4., 3, 2, 1], [1., 4, 3, 2], 0.2841)
])
def test_rmsle(pred, target, expected):
    score = rmsle(torch.tensor(pred), torch.tensor(target))
    assert torch.allclose(score, torch.tensor(expected), atol=1e-3)


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param(
        [0., 1., 2., 3.],
        [0., 1., 2., 2.],
        ski_psnr(np.array([0., 1., 2., 3.]), np.array([0., 1., 2., 2.]), data_range=3)
    ),
    pytest.param(
        [4., 3., 2., 1.],
        [1., 4., 3., 2.],
        ski_psnr(np.array([4., 3., 2., 1.]), np.array([1., 4., 3., 2.]), data_range=3)
    )
])
def test_psnr(pred, target, expected):
    score = psnr(pred=torch.tensor(pred),
                 target=torch.tensor(target))
    assert torch.allclose(score, torch.tensor(expected, dtype=torch.float), atol=1e-3)


@pytest.mark.parametrize(['pred', 'target', 'expected'], [
    pytest.param(
        [0., 1., 2., 3.],
        [0., 1., 2., 2.],
        ski_psnr(np.array([0., 1., 2., 3.]), np.array([0., 1., 2., 2.]), data_range=4) * np.log(10)
    ),
    pytest.param(
        [4., 3., 2., 1.],
        [1., 4., 3., 2.],
        ski_psnr(np.array([4., 3., 2., 1.]), np.array([1., 4., 3., 2.]), data_range=4) * np.log(10)
    )
])
def test_psnr_base_e_wider_range(pred, target, expected):
    score = psnr(pred=torch.tensor(pred),
                 target=torch.tensor(target),
                 data_range=4,
                 base=2.718281828459045)
    assert torch.allclose(score, torch.tensor(expected, dtype=torch.float32), atol=1e-3)


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    pytest.param(ski_psnr, psnr, id='peak_signal_noise_ratio')
])
def test_psnr_against_sklearn(sklearn_metric, torch_metric):
    """Compare PL metrics to sklearn version."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred = torch.randint(10, (500,), device=device, dtype=torch.float)
    target = torch.randint(10, (500,), device=device, dtype=torch.float)

    sk_score = sklearn_metric(target.cpu().detach().numpy(),
                              pred.cpu().detach().numpy(),
                              data_range=10)
    sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
    th_score = torch_metric(pred, target, data_range=10)

    pred = torch.randint(5, (500,), device=device, dtype=torch.float)
    target = torch.randint(10, (500,), device=device, dtype=torch.float)

    sk_score = sklearn_metric(target.cpu().detach().numpy(),
                              pred.cpu().detach().numpy(),
                              data_range=10)
    sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
    th_score = torch_metric(pred, target, data_range=10)
    assert torch.allclose(sk_score, th_score)

    pred = torch.randint(10, (500,), device=device, dtype=torch.float)
    target = torch.randint(5, (500,), device=device, dtype=torch.float)

    sk_score = sklearn_metric(target.cpu().detach().numpy(),
                              pred.cpu().detach().numpy(),
                              data_range=5)
    sk_score = torch.tensor(sk_score, dtype=torch.float, device=device)
    th_score = torch_metric(pred, target, data_range=5)
    assert torch.allclose(sk_score, th_score)

