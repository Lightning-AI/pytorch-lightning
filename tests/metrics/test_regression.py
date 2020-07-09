# NOTE: This file only tests if modules with arguments are running fine.
#   The actual metric implementation is tested in functional/test_regression.py
#   Especially reduction and reducing across processes won't be tested here!

import torch
from skimage.metrics import peak_signal_noise_ratio as ski_psnr

from pytorch_lightning.metrics.regression import (
    MAE, MSE, RMSE, RMSLE, PSNR
)


def test_mse():
    mse = MSE()
    assert mse.name == 'mse'

    pred = torch.tensor([0., 1, 2, 3])
    target = torch.tensor([0., 1, 2, 2])
    score = mse(pred, target)
    assert isinstance(score, torch.Tensor)


def test_rmse():
    rmse = RMSE()
    assert rmse.name == 'rmse'

    pred = torch.tensor([0., 1, 2, 3])
    target = torch.tensor([0., 1, 2, 2])
    score = rmse(pred, target)
    assert isinstance(score, torch.Tensor)


def test_mae():
    mae = MAE()
    assert mae.name == 'mae'

    pred = torch.tensor([0., 1, 2, 3])
    target = torch.tensor([0., 1, 2, 2])
    score = mae(pred, target)
    assert isinstance(score, torch.Tensor)


def test_rmsle():
    rmsle = RMSLE()
    assert rmsle.name == 'rmsle'

    pred = torch.tensor([0., 1, 2, 3])
    target = torch.tensor([0., 1, 2, 2])
    score = rmsle(pred, target)
    assert isinstance(score, torch.Tensor)


def test_psnr():
    psnr = PSNR()
    assert psnr.name == 'psnr'

    pred = torch.tensor([0., 1, 2, 3])
    target = torch.tensor([0., 1, 2, 2])
    score = psnr(pred, target)
    assert isinstance(score, torch.Tensor)
