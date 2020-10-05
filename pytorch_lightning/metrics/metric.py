import functools
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Union
from collections.abc import Mapping, Sequence
from collections import namedtuple
from copy import deepcopy

import os
import torch
from torch import nn

from pytorch_lightning.utilities.apply_func import apply_to_collection


class Metric(nn.Module, ABC):

    def __init__(
        self,
        compute_on_step: bool = True,
        ddp_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
    ):
        super().__init__()

        self.ddp_sync_on_step = ddp_sync_on_step
        self.compute_on_step = compute_on_step
        self.process_group = process_group
        self._to_sync = True

        self.update = self.wrap_update(self.update)
        self.compute = self.wrap_compute(self.compute)
        self._computed = None

        # initialize state
        self._reductions = {}
        self._defaults = {}

    def add_state(self, name, default, reduction=None):
        if reduction is None:
            # TODO: implement default reduction
            raise NotImplementedError("default reduction not implemented")

        setattr(self, name, default)
        self._defaults[name] = deepcopy(default)
        self._reductions[name] = reduction

    def forward(self, *args, **kwargs):
        # add current step
        self.update(*args, **kwargs)

        if self.compute_on_step:
            self._to_sync = self.ddp_sync_on_step

            # save context before switch
            self._cache = {attr: getattr(self, attr) for attr in self._defaults.keys()}

            # call reset, update, compute, on single batch
            self.reset()
            self.update(*args, **kwargs)
            result = self.compute()

            # restore context
            for attr, val in self._cache.items():
                setattr(self, attr, val)
            self._to_sync = True
            self._computed = None

            return result

    def sync(self):
        input_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
        output_dict = apply_to_collection(
            input_dict,
            torch.Tensor,
            gather_all_tensors_if_available,
            group=self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # agregate lists of tensors
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)
        
    def wrap_update(self, update):
        @functools.wraps(update)
        def wrapped_func(*args, **kwargs):
            self._computed = None
            return update(*args, **kwargs)
        return wrapped_func

    def wrap_compute(self, compute):
        @functools.wraps(compute)
        def wrapped_func(*args, **kwargs):
            # return cached value
            if self._computed is not None:
                return self._computed

            if self._to_sync and torch.distributed.is_available() and torch.distributed.is_initialized():
                self.sync()

            self._computed = compute(*args, **kwargs)
            self.reset()

            return self._computed
            
        return wrapped_func

    @abstractmethod
    def update(self) -> None:  # pylint: disable=E0202
        pass

    @abstractmethod
    def compute(self):  # pylint: disable=E0202
        pass

    def reset(self):
        for attr, default in self._defaults.items():
            setattr(self, attr, deepcopy(default))


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
