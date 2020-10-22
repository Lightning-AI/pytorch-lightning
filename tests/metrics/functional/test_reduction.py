import pytest
import torch

from pytorch_lightning.metrics.functional.reduction import reduce, class_reduce


def test_reduce():
    start_tensor = torch.rand(50, 40, 30)

    assert torch.allclose(reduce(start_tensor, 'elementwise_mean'), torch.mean(start_tensor))
    assert torch.allclose(reduce(start_tensor, 'sum'), torch.sum(start_tensor))
    assert torch.allclose(reduce(start_tensor, 'none'), start_tensor)

    with pytest.raises(ValueError):
        reduce(start_tensor, 'error_reduction')


def test_class_reduce():
    num = torch.randint(1, 10, (100,)).float()
    denom = torch.randint(10, 20, (100,)).float()
    weights = torch.randint(1, 100, (100,)).float()

    assert torch.allclose(class_reduce(num, denom, weights, 'micro'),
                          torch.sum(num) / torch.sum(denom))
    assert torch.allclose(class_reduce(num, denom, weights, 'macro'),
                          torch.mean(num / denom))
    assert torch.allclose(class_reduce(num, denom, weights, 'weighted'),
                          torch.sum(num / denom * (weights / torch.sum(weights))))
    assert torch.allclose(class_reduce(num, denom, weights, 'none'),
                          num / denom)
