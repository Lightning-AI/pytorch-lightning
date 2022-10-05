import datetime
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch
from torch.distributed import Backend, ProcessGroup, ReduceOp

_NO_GROUP_PASSED = -1


class Collective(ABC):
    def __init__(self, groups: Optional[Union[List[ProcessGroup], Dict[str, ProcessGroup]]] = None):
        self.managed_groups = set()
        self.current_group = "default"
        self.groups: Dict[str, Optional[ProcessGroup]] = dict()
        if isinstance(groups, list):
            for i, pg in enumerate(groups):
                self.groups[str(i)] = pg
        elif isinstance(groups, dict):
            for key in groups:
                self.groups[str(key)] = groups[key]

        self.groups["default"] = None

    def pass_group_to_fn(fn: Callable) -> Callable:
        @wraps(fn)
        def inner(self, *args, **kwargs):
            params = inspect.signature(fn).parameters
            group_index = params.values().index("group")
            if len(args) <= group_index and "group" not in kwargs:  # "group" is not passed
                kwargs["group"] = self.groups[self.current_group]
            return fn(*args, **kwargs)

        return inner

    def get_available_groups(self) -> List[str]:
        return list(self.groups.keys())

    @contextmanager
    def use_process_group(self, group: Optional[str] = None):
        if group is None:
            group = "default"
        if group not in self.groups:
            raise ValueError(f"Process group {group} not found. Available: {self.get_available_groups()}")
        old_group, self.current_group = self.current_group, group
        yield
        self.current_group = old_group

    def init_group(
        self,
        backend: Union[str, Backend],
        world_size: int = -1,
        rank: int = -1,
        group_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if group_name is None:
            group_name = f"{str(backend)}_{rank}_{world_size}"
        if group_name in self.groups:
            raise ValueError(f"Process group {group_name} already exists")
        group = self._init_group_impl(backend, world_size, rank, group_name, **kwargs)
        if group is not None:
            self.groups[group_name] = group
            self.managed_groups.add(group_name)

    @abstractmethod
    def _init_group_impl(
        self,
        backend: Union[str, Backend],
        world_size: int = -1,
        rank: int = -1,
        group_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[ProcessGroup]:
        pass

    @pass_group_to_fn
    @abstractmethod
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @pass_group_to_fn
    @abstractmethod
    def broadcast_object_list(
        self,
        object_list: List[Any],
        src: int,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        device: Optional[torch.device] = None,
    ) -> List[Any]:
        pass

    @pass_group_to_fn
    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @pass_group_to_fn
    @abstractmethod
    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
        op: ReduceOp = ReduceOp.SUM,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @pass_group_to_fn
    @abstractmethod
    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op: bool = False,
    ) -> List[torch.Tensor]:
        pass

    @pass_group_to_fn
    @abstractmethod
    def all_gather_object(
        self,
        object_list: List[Any],
        object: Any,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
    ) -> List[Any]:
        pass

    @pass_group_to_fn
    @abstractmethod
    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        dst: int = 0,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op: bool = False,
    ) -> List[torch.Tensor]:
        pass

    @pass_group_to_fn
    @abstractmethod
    def gather_object(
        self,
        obj: Any,
        object_gather_list: Optional[List[Any]] = None,
        dst: int = 0,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
    ) -> List[Any]:
        pass

    @pass_group_to_fn
    @abstractmethod
    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @pass_group_to_fn
    @abstractmethod
    def scatter_object_list(
        self,
        scatter_object_output_list: List[Any],
        scatter_object_input_list: Optional[List[Any]],
        src: int = 0,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
    ) -> List[Any]:
        pass

    @pass_group_to_fn
    @abstractmethod
    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: ReduceOp = ReduceOp.SUM,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op=False,
    ) -> torch.Tensor:
        pass

    @pass_group_to_fn
    @abstractmethod
    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op: bool = False,
    ) -> List[torch.Tensor]:
        pass

    @pass_group_to_fn
    @abstractmethod
    def barrier(
        self,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        async_op: bool = False,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        pass

    @pass_group_to_fn
    @abstractmethod
    def monitored_barrier(
        self,
        group: Optional[Union[ProcessGroup, Literal[_NO_GROUP_PASSED]]] = _NO_GROUP_PASSED,
        timeout: Optional[datetime.timedelta] = None,
        wait_all_ranks: bool = False,
    ) -> None:
        pass
