from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from typing_extensions import Self

from lightning_lite.utilities.types import CollectibleGroup


class Collective(ABC):
    def __init__(self, instantiate_group: bool = False, **group_kwargs: Any) -> None:
        self._group_kwargs = group_kwargs
        self._group: Optional[CollectibleGroup] = None
        if instantiate_group:
            self.create_group()

    def create_group(self, **kwargs: Any) -> Self:  # type: ignore[valid-type]
        if self._group is not None:
            raise RuntimeError(f"{type(self).__name__} already owns a group.")
        self._group_kwargs.update(kwargs)
        self._group = self.init_group(**self._group_kwargs)
        return self

    @property
    def group(self) -> CollectibleGroup:
        if self._group is None:
            raise RuntimeError(
                f"{type(self).__name__} does not own a group. HINT: try `collective.create_group().group`"
            )
        return self._group

    @property
    @abstractmethod
    def rank(self) -> int:
        pass

    @property
    @abstractmethod
    def world_size(self) -> int:
        pass

    @staticmethod
    @abstractmethod
    def init_group(
        **kwargs: Any,
    ) -> CollectibleGroup:
        pass

    def teardown(self) -> None:
        if self._group is None:
            raise RuntimeError(f"{type(self).__name__} does not own a group to destroy.")
        self.destroy_group(self._group)
        self._group = None

    @staticmethod
    @abstractmethod
    def destroy_group(group: CollectibleGroup) -> None:
        pass

    @staticmethod
    @abstractmethod
    def _convert_to_native_op(op: str) -> Any:
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
        op: str,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
        op: str,
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
    ) -> Optional[List[torch.Tensor]]:
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
        op: str,
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
