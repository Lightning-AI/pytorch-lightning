import pytest
import torch

from skimage.metrics import peak_signal_noise_ratio as ski_psnr
from pytorch_lightning.metrics.functional.regression import psnr


@pytest.mark.parametrize(['sklearn_metric', 'torch_metric'], [
    pytest.param(ski_psnr, psnr, id='peak_signal_noise_ratio')
])
def test_psnr_against_sklearn(sklearn_metric, torch_metric):
    """Compare PL metrics to sklearn version."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred = torch.randint(10, (500,), device=device, dtype=torch.double)
    target = torch.randint(10, (500,), device=device, dtype=torch.double)
    assert torch.allclose(
        torch.tensor(sklearn_metric(target.cpu().detach().numpy(),
                                    pred.cpu().detach().numpy(),
                                    data_range=10), dtype=torch.double, device=device),
        torch_metric(pred, target, data_range=10))

    pred = torch.randint(5, (500,), device=device, dtype=torch.double)
    target = torch.randint(10, (500,), device=device, dtype=torch.double)
    assert torch.allclose(
        torch.tensor(sklearn_metric(target.cpu().detach().numpy(),
                                    pred.cpu().detach().numpy(),
                                    data_range=10), dtype=torch.double, device=device),
        torch_metric(pred, target, data_range=10))

    pred = torch.randint(10, (500,), device=device, dtype=torch.double)
    target = torch.randint(5, (500,), device=device, dtype=torch.double)
    assert torch.allclose(
        torch.tensor(sklearn_metric(target.cpu().detach().numpy(),
                                    pred.cpu().detach().numpy(),
                                    data_range=5), dtype=torch.double, device=device),
        torch_metric(pred, target, data_range=5))
