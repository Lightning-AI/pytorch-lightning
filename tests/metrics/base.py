from collections import namedtuple

import numpy as np
import pytest
import torch
import torch.distributed as dist

import tests.base.utils as tutils
from pytorch_lightning.metrics.metric import _sync_ddp, _sync_collections, BaseMetric


@pytest.mark.skipif(torch.cuda.device_count() < 2, "test requires multi-GPU machine")
def test_sync_reduce_ddp():
    """Make sure sync-reduce works with DDP"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    dist.init_process_group('gloo')

    tensor = torch.tensor([1.], device='cuda:0')

    reduced_tensor = _sync_ddp(tensor)
    assert reduced_tensor.item() == dist.get_world_size(), \
        'Sync-Reduce does not work properly with DDP and Tensors'

    number = 1.
    reduced_number = _sync_ddp(number)
    assert isinstance(reduced_number, torch.Tensor), 'When reducing a number we should get a tensor out'
    assert reduced_number.item() == dist.get_world_size(), \
        'Sync-Reduce does not work properly with DDP and Numbers'

    dist.destroy_process_group()


def test_sync_reduce_simple():
    """Make sure sync-reduce works without DDP"""
    tensor = torch.tensor([1.], device='cpu')

    reduced_tensor = _sync_ddp(tensor)

    assert torch.allclose(tensor,
                          reduced_tensor), 'Sync-Reduce does not work properly without DDP and Tensors'

    number = 1.
    reduced_number = _sync_ddp(number)
    assert isinstance(reduced_number, torch.Tensor), 'When reducing a number we should get a tensor out'
    assert reduced_number.item() == number, 'Sync-Reduce does not work properly without DDP and Numbers'


def _sync_collections_test(is_ddp: bool):
    ntc = namedtuple('Foo', ['bar'])
    to_reduce = {
        'a': torch.tensor([1.]),  # Tensor
        'b': [torch.tensor([2.])],  # list
        'c': (torch.tensor([100.]),),  # tuple
        'd': ntc(bar=5.),  # named tuple
        'e': np.array([10.]),  # numpy array
        'f': 'this_is_a_dummy_str',  # string
        'g': 12.  # number
    }

    if is_ddp:
        factor = dist.get_world_size()
    else:
        factor = 1.

    expected_result = {
        'a': torch.tensor([1. * factor]),
        'b': [torch.tensor([2. * factor])],
        'c': (torch.tensor([100. * factor]),),
        'd': ntc(bar=torch.tensor([5. * factor])),
        'e': torch.tensor([10. * factor]),
        'f': 'this_is_a_dummy_str',
        'g': torch.tensor([12. * factor]),
    }

    reduced = _sync_collections(to_reduce)

    assert isinstance(reduced, dict), ' Type Consistency of dict not preserved'
    assert all([x in reduced for x in to_reduce.keys()]), 'Not all entries of the dict were preserved'
    assert all([isinstance(reduced[k], type(expected_result[k])) for k in to_reduce.keys()]), \
        'At least one type was not correctly preserved'

    assert isinstance(reduced['a'], torch.Tensor), 'Reduction Result of a Tensor should be a Tensor'
    assert torch.allclose(expected_result['a'],
                          reduced['a']), 'Reduction of a tensor does not yield the expected value'

    assert isinstance(reduced['b'], list), 'Reduction Result of a list should be a list'
    assert all([torch.allclose(x, y) for x, y in zip(reduced['b'], expected_result['b'])]), \
        'At least one value of list reduction did not come out as expected'

    assert isinstance(reduced['c'], tuple), 'Reduction Result of a tuple should be a tuple'
    assert all([torch.allclose(x, y) for x, y in zip(reduced['c'], expected_result['c'])]), \
        'At least one value of tuple reduction did not come out as expected'

    assert isinstance(reduced['d'], ntc), 'Type Consistency for named tuple not given'
    assert isinstance(reduced['d'].bar,
                      torch.Tensor), 'Failure in type promotion while reducing fields of named tuples'
    assert torch.allclose(reduced['d'].bar, expected_result['d'].bar)

    assert isinstance(reduced['e'], torch.Tensor), 'Type Promotion in reduction of numpy arrays failed'
    assert torch.allclose(reduced['e'], expected_result['e']), \
        'Reduction of numpy array did not yield the expected result'

    assert isinstance(reduced['f'], str), 'A string should not be reduced'
    assert reduced['f'] == expected_result['f'], 'String not preserved during reduction'

    assert isinstance(reduced['g'], torch.Tensor), 'Reduction of a number should result in a tensor'
    assert torch.allclose(reduced['g'],
                          expected_result['g']), 'Reduction of a number did not yield the desired result'


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    'Not enough GPUs to test sync reduce')
def test_sync_collections_ddp():
    tutils.reset_seed()
    tutils.set_random_master_port()

    dist.init_process_group('gloo')

    _sync_collections_test(True)

    dist.destroy_process_group()


def test_sync_collections_simple():
    _sync_collections_test(False)


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


def test_base_metric_simple():
    _test_base_metric(False)
