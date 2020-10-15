import numpy as np
import pytest
import torch
from skimage.metrics import (
    peak_signal_noise_ratio as ski_psnr,
    structural_similarity as ski_ssim
)

from pytorch_lightning.metrics.functional import (
    psnr,
    ssim
)


@pytest.mark.parametrize(['pred', 'target'], [
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 3.]),
    pytest.param([0., 1., 2., 3.], [0., 1., 2., 2.]),
    pytest.param([4., 3., 2., 1.], [1., 4., 3., 2.]),
])
def test_psnr_with_skimage(pred, target):
    score = psnr(pred=torch.tensor(pred),
                 target=torch.tensor(target), data_range=3)
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


@pytest.mark.parametrize(['size', 'channel', 'coef', 'multichannel'], [
    pytest.param(16, 1, 0.9, False),
    pytest.param(32, 3, 0.8, True),
    pytest.param(48, 4, 0.7, True),
    pytest.param(64, 5, 0.6, True)
])
def test_ssim(size, channel, coef, multichannel):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = torch.rand(size, channel, size, size, device=device)
    target = pred * coef
    ssim_idx = ssim(pred, target, data_range=1.0)
    np_pred = pred.permute(0, 2, 3, 1).cpu().numpy()
    if multichannel is False:
        np_pred = np_pred[:, :, :, 0]
    np_target = np.multiply(np_pred, coef)
    sk_ssim_idx = ski_ssim(
        np_pred, np_target, win_size=11, multichannel=multichannel, gaussian_weights=True, data_range=1.0
    )
    assert torch.allclose(ssim_idx, torch.tensor(sk_ssim_idx, dtype=torch.float, device=device), atol=1e-4)

    ssim_idx = ssim(pred, pred)
    assert torch.allclose(ssim_idx, torch.tensor(1.0, device=device))


@pytest.mark.parametrize(['pred', 'target', 'kernel', 'sigma'], [
    pytest.param([1, 1, 16, 16], [1, 16, 16], [11, 11], [1.5, 1.5]),  # shape
    pytest.param([1, 16, 16], [1, 16, 16], [11, 11], [1.5, 1.5]),  # len(shape)
    pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 11], [1.5]),  # len(kernel), len(sigma)
    pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11], [1.5, 1.5]),  # len(kernel), len(sigma)
    pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11], [1.5]),  # len(kernel), len(sigma)
    pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 0], [1.5, 1.5]),  # invalid kernel input
    pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 10], [1.5, 1.5]),  # invalid kernel input
    pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, -11], [1.5, 1.5]),  # invalid kernel input
    pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 11], [1.5, 0]),  # invalid sigma input
    pytest.param([1, 1, 16, 16], [1, 1, 16, 16], [11, 0], [1.5, -1.5]),  # invalid sigma input
])
def test_ssim_invalid_inputs(pred, target, kernel, sigma):
    pred_t = torch.rand(pred)
    target_t = torch.rand(target, dtype=torch.float64)
    with pytest.raises(TypeError):
        ssim(pred_t, target_t)

    pred = torch.rand(pred)
    target = torch.rand(target)
    with pytest.raises(ValueError):
        ssim(pred, target, kernel, sigma)
