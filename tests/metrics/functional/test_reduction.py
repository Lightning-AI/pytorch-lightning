import pytest
import torch

from pytorch_lightning.metrics.functional.reduction import reduce


def test_reduce():
    start_tensor = torch.rand(50, 40, 30)

    assert torch.allclose(reduce(start_tensor, 'elementwise_mean'), torch.mean(start_tensor))
    assert torch.allclose(reduce(start_tensor, 'sum'), torch.sum(start_tensor))
    assert torch.allclose(reduce(start_tensor, 'none'), start_tensor)

    with pytest.raises(ValueError):
        reduce(start_tensor, 'error_reduction')
