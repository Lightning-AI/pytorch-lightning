import datetime
from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from torch.distributed import ReduceOp


class Collective(ABC):
    def __init__(self, group: Optional[Any] = None, instantiate_pg: bool = False, **pg_kwargs: Any) -> None:
        self._group = group
        self._pg_kwargs = pg_kwargs
        if self._group is None and instantiate_pg:
            self._group = self.init_group(**self._pg_kwargs)

    @property
    def group(self) -> Any:
        if self._group is None:
            self._group = self.init_group(**self._pg_kwargs)
        return self._group

    @property
    @abstractmethod
    def rank(self) -> int:
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        pass

    @abstractmethod
    def init_group(
        self,
        **init_kwargs: Any,
    ) -> Any:
        pass

    @abstractmethod
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def broadcast_object_list(
        self,
        object_list: List[Any],
        src: int,
        device: Optional[torch.device] = None,
    ) -> List[Any]:
        pass

    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
        op: ReduceOp = ReduceOp.SUM,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        async_op: bool = False,
    ) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def all_gather_object(
        self,
        object_list: List[Any],
        object: Any,
    ) -> List[Any]:
        pass

    @abstractmethod
    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        dst: int = 0,
        async_op: bool = False,
    ) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def gather_object(
        self,
        obj: Any,
        object_gather_list: Optional[List[Any]] = None,
        dst: int = 0,
    ) -> List[Any]:
        pass

    @abstractmethod
    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def scatter_object_list(
        self,
        scatter_object_output_list: List[Any],
        scatter_object_input_list: Optional[List[Any]],
        src: int = 0,
    ) -> List[Any]:
        pass

    @abstractmethod
    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: ReduceOp = ReduceOp.SUM,
        async_op: bool = False,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
        async_op: bool = False,
    ) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def barrier(
        self,
        async_op: bool = False,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        pass

    @abstractmethod
    def monitored_barrier(
        self,
        timeout: Optional[datetime.timedelta] = None,
        wait_all_ranks: bool = False,
    ) -> None:
        pass
