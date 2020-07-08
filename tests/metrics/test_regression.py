import pytest
import torch
from skimage.metrics import peak_signal_noise_ratio as ski_psnr
import numpy as np

from pytorch_lightning.metrics.regression import (
    MAE, MSE, RMSE, RMSLE, PSNR
)


@pytest.mark.parametrize(['pred', 'target', 'exp'], [
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 2.], .25),
    pytest.param([4., 3., 2., 1.], [1., 4., 3., 2.], 3.)
])
def test_mse(pred, target, exp):
    mse = MSE()
    assert mse.name == 'mse'

    score = mse(pred=torch.tensor(pred),
                target=torch.tensor(target))

    assert isinstance(score, torch.Tensor)
    assert score.item() == exp


@pytest.mark.parametrize(['pred', 'target', 'exp'], [
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 2.], .5),
    pytest.param([4., 3., 2., 1.], [1., 4., 3., 2.], 1.7321)
])
def test_rmse(pred, target, exp):
    rmse = RMSE()
    assert rmse.name == 'rmse'

    score = rmse(pred=torch.tensor(pred),
                 target=torch.tensor(target))

    assert isinstance(score, torch.Tensor)
    assert pytest.approx(score.item(), rel=1e-3) == exp


@pytest.mark.parametrize(['pred', 'target', 'exp'], [
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 2.], .25),
    pytest.param([4., 3., 2., 1.], [1., 4., 3., 2.], 1.5)
])
def test_mae(pred, target, exp):
    mae = MAE()
    assert mae.name == 'mae'

    score = mae(pred=torch.tensor(pred),
                target=torch.tensor(target))

    assert isinstance(score, torch.Tensor)
    assert score.item() == exp


@pytest.mark.parametrize(['pred', 'target', 'exp'], [
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 2.], .0207),
    pytest.param([4., 3., 2., 1.], [1., 4., 3., 2.], .2841)
])
def test_rmsle(pred, target, exp):
    rmsle = RMSLE()
    assert rmsle.name == 'rmsle'

    score = rmsle(pred=torch.tensor(pred),
                  target=torch.tensor(target))

    assert isinstance(score, torch.Tensor)
    assert pytest.approx(score.item(), rel=1e-3) == exp


@pytest.mark.parametrize(['pred', 'target', 'exp'], [
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
def test_psnr(pred, target, exp):
    psnr = PSNR()
    assert psnr.name == 'psnr'
    score = psnr(pred=torch.tensor(pred),
                 target=torch.tensor(target))
    assert isinstance(score, torch.Tensor)
    assert pytest.approx(score.item(), rel=1e-3) == exp


@pytest.mark.parametrize(['pred', 'target', 'exp'], [
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
def test_psnr_base_e_wider_range(pred, target, exp):
    psnr = PSNR(data_range=4, base=2.718281828459045)
    assert psnr.name == 'psnr'
    score = psnr(pred=torch.tensor(pred),
                 target=torch.tensor(target))
    assert isinstance(score, torch.Tensor)
    assert pytest.approx(score.item(), rel=1e-3) == exp
