from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from torch.distributed import ReduceOp
from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class CollectibleGroup(Protocol):
    def size(self) -> int:
        ...

    def rank(self) -> int:
        ...


class Collective(ABC):
    def __init__(self, instantiate_group: bool = False, **group_kwargs: Any) -> None:
        self._group: Optional[CollectibleGroup] = None
        self._group_kwargs = group_kwargs
        if instantiate_group:
            _ = self.group

    @property
    def group(self) -> CollectibleGroup:
        if self._group is None:
            self._group = self.init_group(**self._group_kwargs)
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
    ) -> CollectibleGroup:
        pass

    @abstractmethod
    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
        op: ReduceOp = ReduceOp.SUM,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        dst: int = 0,
    ) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: ReduceOp = ReduceOp.SUM,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        pass

    @abstractmethod
    def barrier(
        self,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        pass
