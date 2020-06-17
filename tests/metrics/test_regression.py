import pytest
import torch

from pytorch_lightning.metrics.regression import (
    MAE, MSE, RMSE, RMSLE
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
