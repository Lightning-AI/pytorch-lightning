import pytest
import torch
import torch.distributed as dist

import tests.base.utils as tutils
from pytorch_lightning.metrics.metric import _sync_ddp_to_device_type, BaseMetric


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    'Not enough GPUs to test sync reduce')
def test_sync_reduce_ddp():
    """Make sure sync-reduce works with DDP"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    dist.init_process_group('gloo')

    tensor = torch.tensor([1.], device='cuda:0')

    reduced_tensor = _sync_ddp_to_device_type(tensor, device='cpu', dtype=torch.float)
    assert reduced_tensor.item() == dist.get_world_size(), \
        'Sync-Reduce does not work properly with DDP and Tensors'
    assert reduced_tensor.device == torch.device('cpu'), 'Reduced Tensor was not pusehd to correct device'
    assert reduced_tensor.dtype == torch.float, 'Reduced Tensor was not converted to correct dtype'

    number = 1.
    reduced_number = _sync_ddp_to_device_type(number, device='cpu', dtype=torch.float)
    assert isinstance(reduced_number, torch.Tensor), 'When reducing a number we should get a tensor out'
    assert reduced_number.item() == dist.get_world_size(), \
        'Sync-Reduce does not work properly with DDP and Numbers'
    assert reduced_number.device == torch.device('cpu'), 'Reduced Tensor was not pusehd to correct device'
    assert reduced_number.dtype == torch.float, 'Reduced Tensor was not converted to correct dtype'

    dist.destroy_process_group()


def test_sync_reduce_no_ddp():
    """Make sure sync-reduce works without DDP"""
    tensor = torch.tensor([1.], device='cpu')

    reduced_tensor = _sync_ddp_to_device_type(tensor, device='cpu', dtype=torch.float)

    assert torch.allclose(tensor,
                          reduced_tensor), 'Sync-Reduce does not work properly without DDP and Tensors'

    number = 1.
    reduced_number = _sync_ddp_to_device_type(number, device='cpu', dtype=torch.float)
    assert isinstance(reduced_number, torch.Tensor), 'When reducing a number we should get a tensor out'
    assert reduced_number.item() == number, 'Sync-Reduce does not work properly without DDP and Numbers'


def _test_base_metric(is_ddp):
    class DummyMetric(BaseMetric):
        def __init__(self):
            super().__init__(name='Dummy')

        def forward(self):
            return 1.

    dummy_metric = DummyMetric()

    assert dummy_metric.name == 'Dummy'
    metric_val = dummy_metric()

    if is_ddp:
        expected = dist.get_world_size()
    else:
        expected = 1.

    assert isinstance(metric_val, torch.Tensor), \
        'The result value should be synced and reduced which would promote the type from number to tensor'
    assert metric_val.item() == expected, 'Invalid Value for reduction'


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    'Not enough GPUs to test with ddp')
def test_base_metric_ddp():
    _test_base_metric(True)


def test_base_metric_no_ddp():
    _test_base_metric(False)
