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
from pytorch_lightning.metrics.utils import _flatten, gather_all_tensors_if_available


class Metric(nn.Module, ABC):
    """

    """
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

        self.update = self._wrap_update(self.update)
        self.compute = self._wrap_compute(self.compute)
        self._computed = None

        # initialize state
        self._reductions = {}
        self._defaults = {}

    def add_state(self, name: str, default, dist_reduce_fx: Optional[Union[str, Callable]] = None):
        """
        Adds metric state variable. Only used by subclasses.

        Args:
            name: The name of the state variable. The variable will then be accessible at ``self.name``.
            default: Default value of the state; can either be a tensor or an empty list. The state will be
                reset to this value when ``self.reset()`` is called.
            dist_reduce_fx (Optional): Function to reduce state accross mutliple GPUs. If value is ``"sum"``,
                ``"mean"``, or ``"cat"``, we will use ``torch.sum``, ``torch.mean``, and ``torch.cat`` respectively,
                each with argument ``dim=0``.
        """
        if not isinstance(default, torch.Tensor) or (isinstance(default, list) and len(default) != 0):
            raise ValueError(
                "state variable must be a tensor or any empty list (where you can append tensors)"
            )

        if dist_reduce_fx == "sum":
            dist_reduce_fx = lambda x: torch.sum(x, dim=0)
        elif dist_reduce_fx == "mean":
            dist_reduce_fx = lambda x: torch.mean(x, dim=0)
        elif dist_reduce_fx == "cat":
            dist_reduce_fx = lambda x: torch.cat(x, dim=0)
        elif dist_reduce_fx is not None and not isinstance(dist_reduce_fx, Callable):
            raise ValueError(
                "`dist_reduce_fx` must be callable or one of ['mean', 'sum', 'cat', None]"
            )

        setattr(self, name, default)
        self._defaults[name] = deepcopy(default)
        self._reductions[name] = dist_reduce_fx

    def forward(self, *args, **kwargs):
        """
        """
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

    def _sync_dist(self):
        """
        Method to synchronize metric state variables across different processes
        in distributed training.
        """
        input_dict = {attr: getattr(self, attr) for attr in self._reductions.keys()}
        output_dict = apply_to_collection(
            input_dict,
            torch.Tensor,
            gather_all_tensors_if_available,
            group=self.process_group,
        )

        for attr, reduction_fn in self._reductions.items():
            # pre-processing ops (stack or flatten for inputs)
            if isinstance(output_dict[attr][0], torch.Tensor):
                output_dict[attr] = torch.stack(output_dict[attr])
            elif isinstance(output_dict[attr][0], list):
                output_dict[attr] = _flatten(output_dict[attr])

            assert isinstance(reduction_fn, (Callable, None))
            reduced = reduction_fn(output_dict[attr]) if reduction_fn is not None else output_dict[attr]
            setattr(self, attr, reduced)

    def _wrap_update(self, update):
        @functools.wraps(update)
        def wrapped_func(*args, **kwargs):
            self._computed = None
            return update(*args, **kwargs)
        return wrapped_func

    def _wrap_compute(self, compute):
        @functools.wraps(compute)
        def wrapped_func(*args, **kwargs):
            # return cached value
            if self._computed is not None:
                return self._computed

            if self._to_sync \
                and torch.distributed.is_available() \
                and torch.distributed.is_initialized():
                self._sync_dist()

            self._computed = compute(*args, **kwargs)
            self.reset()

            return self._computed

        return wrapped_func

    @abstractmethod
    def update(self) -> None:  # pylint: disable=E0202
        """
        Override this method to update the state variables of your metric class.
        """
        pass

    @abstractmethod
    def compute(self):  # pylint: disable=E0202
        """
        Override this method to compute the final metric value from state variables
        synchronized across the distributed backend.
        """
        pass

    def reset(self):
        """
        This method automatically resets the metric state variables to their default value.
        """
        for attr, default in self._defaults.items():
            setattr(self, attr, deepcopy(default))
