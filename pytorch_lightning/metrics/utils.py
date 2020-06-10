import numbers
from typing import Union, Any, Optional

import numpy as np
import torch
from torch.utils.data._utils.collate import default_convert

from pytorch_lightning.utilities.apply_func import apply_to_collection


def _apply_to_inputs(func_to_apply, *dec_args, **dec_kwargs):
    def decorator_fn(func_to_decorate):
        def new_func(*args, **kwargs):
            args = func_to_apply(args, *dec_args, **dec_kwargs)
            kwargs = func_to_apply(kwargs, *dec_args, **dec_kwargs)
            return func_to_decorate(*args, **kwargs)

        return new_func

    return decorator_fn


def _apply_to_outputs(func_to_apply, *dec_args, **dec_kwargs):
    def decorator_fn(function_to_decorate):
        def new_func(*args, **kwargs):
            result = function_to_decorate(*args, **kwargs)
            return func_to_apply(result, *dec_args, **dec_kwargs)

        return new_func

    return decorator_fn


def _convert_to_tensor(data: Any) -> Any:
    """
    Maps all kind of collections and numbers to tensors

    Args:
        data: the data to convert to tensor

    Returns:
        the converted data

    """
    if isinstance(data, numbers.Number):
        return torch.tensor([data])
    else:
        return default_convert(data)


def _convert_to_numpy(data: Union[torch.Tensor, np.ndarray, numbers.Number]) -> np.ndarray:
    """
    converts all tensors and numpy arrays to numpy arrays
    Args:
        data: the tensor or array to convert to numpy

    Returns:
        the resulting numpy array

    """
    if isinstance(data, torch.Tensor):
        return data.cpu().detach().numpy()
    elif isinstance(data, numbers.Number):
        return np.array([data])
    return data


def _numpy_metric_conversion(func_to_decorate):
    # Applies collection conversion from tensor to numpy to all inputs
    # we need to include numpy arrays here, since otherwise they will also be treated as sequences
    func_convert_inputs = _apply_to_inputs(
        apply_to_collection, (torch.Tensor, np.ndarray, numbers.Number), _convert_to_numpy)(func_to_decorate)
    # converts all inputs back to tensors (device doesn't matter here, since this is handled by BaseMetric)
    func_convert_in_out = _apply_to_outputs(_convert_to_tensor)(func_convert_inputs)
    return func_convert_in_out


def _tensor_metric_conversion(func_to_decorate):
    # Converts all inputs to tensor if possible
    func_convert_inputs = _apply_to_inputs(_convert_to_tensor)(func_to_decorate)
    # convert all outputs to tensor if possible
    return _apply_to_outputs(_convert_to_tensor)(func_convert_inputs)


def _sync_ddp(result: Union[torch.Tensor],
              group: Any = torch.distributed.group.WORLD,
              reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM,
              ) -> torch.Tensor:
    """
    Function to reduce the tensors from several ddp processes to one master process

    Args:
        result: the value to sync and reduce (typically tensor or number)
        device: the device to put the synced and reduced value to
        dtype: the datatype to convert the synced and reduced value to
        group: the process group to gather results from. Defaults to all processes (world)
        reduce_op: the reduction operation. Defaults to sum

    Returns:
        reduced value

    """

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # sync all processes before reduction
        torch.distributed.barrier(group=group)
        torch.distributed.all_reduce(result, op=reduce_op, group=group,
                                     async_op=False)

    return result


def numpy_metric(group: Any = torch.distributed.group.WORLD,
                 reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM):
    def decorator_fn(func_to_decorate):
        return _apply_to_outputs(apply_to_collection, torch.Tensor, _sync_ddp,
                                 group=group,
                                 reduce_op=reduce_op)(_numpy_metric_conversion(func_to_decorate))

    return decorator_fn


def tensor_metric(group: Any = torch.distributed.group.WORLD,
                  reduce_op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM):
    def decorator_fn(func_to_decorate):
        return _apply_to_outputs(apply_to_collection, torch.Tensor, _sync_ddp,
                                 group=group,
                                 reduce_op=reduce_op)(_tensor_metric_conversion(func_to_decorate))

    return decorator_fn
