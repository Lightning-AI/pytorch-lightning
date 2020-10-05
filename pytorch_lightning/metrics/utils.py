import torch

from typing import Any, Callable, Optional, Union


def _flatten(x):
    return [item for sublist in x for item in sublist]


def gather_all_tensors_if_available(result: Union[torch.Tensor], group: Optional[Any] = None):
    """
    Function to gather all tensors from several ddp processes onto a list that
    is broadcasted to all processes

    Args:
        result: the value to sync
        group: the process group to gather results from. Defaults to all processes (world)

    Return:
        gathered_result: list with size equal to the process group where
            gathered_result[i] corresponds to result tensor from process i

    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD

        world_size = torch.distributed.get_world_size(group)

        gathered_result = [torch.zeros_like(result) for _ in range(world_size)]

        # sync and broadcast all
        torch.distributed.barrier(group=group)
        torch.distributed.all_gather(gathered_result, result, group)

        result = gathered_result
    return result
