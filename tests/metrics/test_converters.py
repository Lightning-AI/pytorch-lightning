import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tests.base.utils as tutils
from pytorch_lightning.metrics.converters import _apply_to_inputs, _apply_to_outputs, \
    _convert_to_tensor, _convert_to_numpy, _numpy_metric_conversion, \
    _tensor_metric_conversion, _sync_ddp_if_available, tensor_metric, numpy_metric


def test_apply_to_inputs():
    def apply_fn(inputs, factor):
        if isinstance(inputs, (float, int)):
            return inputs * factor
        elif isinstance(inputs, dict):
            return {k: apply_fn(v, factor) for k, v in inputs.items()}
        elif isinstance(inputs, (tuple, list)):
            return [apply_fn(x, factor) for x in inputs]

    @_apply_to_inputs(apply_fn, factor=2.)
    def test_fn(*args, **kwargs):
        return args, kwargs

    for args in [[], [1., 2.]]:
        for kwargs in [{}, {'a': 1., 'b': 2.}]:
            result_args, result_kwargs = test_fn(*args, **kwargs)
            assert isinstance(result_args, (list, tuple))
            assert isinstance(result_kwargs, dict)
            assert len(result_args) == len(args)
            assert len(result_kwargs) == len(kwargs)
            assert all([k in result_kwargs for k in kwargs.keys()])
            for arg, result_arg in zip(args, result_args):
                assert arg * 2. == result_arg

            for key in kwargs.keys():
                arg = kwargs[key]
                result_arg = result_kwargs[key]
                assert arg * 2. == result_arg


def test_apply_to_outputs():
    def apply_fn(inputs, additional_str):
        return str(inputs) + additional_str

    @_apply_to_outputs(apply_fn, additional_str='_str')
    def test_fn(*args, **kwargs):
        return 'dummy'

    assert test_fn() == 'dummy_str'


def test_convert_to_tensor():
    for test_item in [1., np.array([1.])]:
        result_tensor = _convert_to_tensor(test_item)
        assert isinstance(result_tensor, torch.Tensor)
        assert result_tensor.item() == 1.


def test_convert_to_numpy():
    for test_item in [1., torch.tensor([1.])]:
        result = _convert_to_numpy(test_item)
        assert isinstance(result, np.ndarray)
        assert result.item() == 1.


def test_numpy_metric_conversion():
    @_numpy_metric_conversion
    def numpy_test_metric(*args, **kwargs):
        for arg in args:
            assert isinstance(arg, np.ndarray)

        for v in kwargs.values():
            assert isinstance(v, np.ndarray)

        return 5.

    result = numpy_test_metric(torch.tensor([1.]), dummy_kwarg=2.)
    assert isinstance(result, torch.Tensor)
    assert result.item() == 5.


def test_tensor_metric_conversion():
    @_tensor_metric_conversion
    def tensor_test_metric(*args, **kwargs):
        for arg in args:
            assert isinstance(arg, torch.Tensor)

        for v in kwargs.values():
            assert isinstance(v, torch.Tensor)

        return 5.

    result = tensor_test_metric(np.array([1.]), dummy_kwarg=2.)
    assert isinstance(result, torch.Tensor)
    assert result.item() == 5.


def setup_ddp(rank, worldsize, ):
    import os

    os.environ['MASTER_ADDR'] = 'localhost'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=worldsize)


def ddp_test_fn(rank, worldsize):
    setup_ddp(rank, worldsize)
    tensor = torch.tensor([1.], device='cuda:0')

    reduced_tensor = _sync_ddp_if_available(tensor)

    assert reduced_tensor.item() == dist.get_world_size(), \
        'Sync-Reduce does not work properly with DDP and Tensors'


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_sync_reduce_ddp():
    """Make sure sync-reduce works with DDP"""
    tutils.reset_seed()
    tutils.set_random_master_port()

    worldsize = 2
    mp.spawn(ddp_test_fn, args=(worldsize,), nprocs=worldsize)

    dist.destroy_process_group()


def test_sync_reduce_simple():
    """Make sure sync-reduce works without DDP"""
    tensor = torch.tensor([1.], device='cpu')

    reduced_tensor = _sync_ddp_if_available(tensor)

    assert torch.allclose(tensor,
                          reduced_tensor), 'Sync-Reduce does not work properly without DDP and Tensors'


def _test_tensor_metric(is_ddp: bool):
    @tensor_metric()
    def tensor_test_metric(*args, **kwargs):
        for arg in args:
            assert isinstance(arg, torch.Tensor)

        for v in kwargs.values():
            assert isinstance(v, torch.Tensor)

        return 5.

    if is_ddp:
        factor = dist.get_world_size()
    else:
        factor = 1.

    result = tensor_test_metric(np.array([1.]), dummy_kwarg=2.)
    assert isinstance(result, torch.Tensor)
    assert result.item() == 5. * factor


def _ddp_test_tensor_metric(rank, worldsize):
    setup_ddp(rank, worldsize)
    _test_tensor_metric(True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_tensor_metric_ddp():
    tutils.reset_seed()
    tutils.set_random_master_port()

    world_size = 2
    mp.spawn(_ddp_test_tensor_metric, args=(world_size,), nprocs=world_size)

    dist.destroy_process_group()


def test_tensor_metric_simple():
    _test_tensor_metric(False)


def _test_numpy_metric(is_ddp: bool):
    @numpy_metric()
    def numpy_test_metric(*args, **kwargs):
        for arg in args:
            assert isinstance(arg, np.ndarray)

        for v in kwargs.values():
            assert isinstance(v, np.ndarray)

        return 5.

    if is_ddp:
        factor = dist.get_world_size()
    else:
        factor = 1.

    result = numpy_test_metric(torch.tensor([1.]), dummy_kwarg=2.)
    assert isinstance(result, torch.Tensor)
    assert result.item() == 5. * factor


def _ddp_test_numpy_metric(rank, worldsize):
    setup_ddp(rank, worldsize)
    _test_numpy_metric(True)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires multi-GPU machine")
def test_numpy_metric_ddp():
    tutils.reset_seed()
    tutils.set_random_master_port()
    world_size = 2
    mp.spawn(_ddp_test_numpy_metric, args=(world_size,), nprocs=world_size)
    dist.destroy_process_group()


def test_numpy_metric_simple():
    _test_numpy_metric(False)
