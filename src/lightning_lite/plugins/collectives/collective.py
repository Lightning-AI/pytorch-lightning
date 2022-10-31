from abc import ABC, abstractmethod
from typing import Any, List, Optional

import torch
from typing_extensions import Self

from lightning_lite.utilities.types import CollectibleGroup


class Collective(ABC):
    """Interface for collective operations.

    Supports communications between multiple processes and multiple nodes. A collective owns a group.

    .. warning::
        This API is experimental and subject to change
    """

    def __init__(self) -> None:
        self._group: Optional[CollectibleGroup] = None

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @property
    @abstractmethod
    def world_size(self) -> int:
        ...

    @property
    def group(self) -> CollectibleGroup:
        if self._group is None:
            raise RuntimeError(
                f"`{type(self).__name__}` does not own a group. HINT: try `collective.create_group().group`"
            )
        return self._group

    @abstractmethod
    def broadcast(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        ...

    @abstractmethod
    def all_reduce(self, tensor: torch.Tensor, op: str) -> torch.Tensor:
        ...

    @abstractmethod
    def reduce(self, tensor: torch.Tensor, dst: int, op: str) -> torch.Tensor:
        ...

    @abstractmethod
    def all_gather(self, tensor_list: List[torch.Tensor], tensor: torch.Tensor) -> List[torch.Tensor]:
        ...

    @abstractmethod
    def gather(self, tensor: torch.Tensor, gather_list: List[torch.Tensor], dst: int = 0) -> List[torch.Tensor]:
        ...

    @abstractmethod
    def scatter(self, tensor: torch.Tensor, scatter_list: List[torch.Tensor], src: int = 0) -> torch.Tensor:
        ...

    @abstractmethod
    def reduce_scatter(self, output: torch.Tensor, input_list: List[torch.Tensor], op: str) -> torch.Tensor:
        ...

    @abstractmethod
    def all_to_all(
        self, output_tensor_list: List[torch.Tensor], input_tensor_list: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        ...

    @abstractmethod
    def send(self, tensor: torch.Tensor, dst: int, tag: Optional[int] = 0) -> None:
        ...

    @abstractmethod
    def recv(self, tensor: torch.Tensor, src: Optional[int] = None, tag: Optional[int] = 0) -> torch.Tensor:
        ...

    @abstractmethod
    def barrier(self, device_ids: Optional[List[int]] = None) -> None:
        ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        ...

    @classmethod
    @abstractmethod
    def is_initialized(cls) -> bool:
        ...

    @classmethod
    @abstractmethod
    def init_group(cls, **kwargs: Any) -> None:
        ...

    @classmethod
    @abstractmethod
    def new_group(cls, **kwargs: Any) -> CollectibleGroup:
        ...

    @classmethod
    @abstractmethod
    def destroy_group(cls, group: CollectibleGroup) -> None:
        ...

    @classmethod
    @abstractmethod
    def _convert_to_native_op(cls, op: str) -> Any:
        ...

    def setup(self, **kwargs: Any) -> Self:  # type: ignore[valid-type]
        if not self.is_initialized():
            self.init_group(**kwargs)
        return self

    def create_group(self, **kwargs: Any) -> Self:  # type: ignore[valid-type]
        """Create a group.

        This assumes that :meth:`~lightning_lite.plugins.collectives.Collective.init_group` has been
        called already by the user.
        """
        if self._group is not None:
            raise RuntimeError(f"`{type(self).__name__}` already owns a group.")
        self._group = self.new_group(**kwargs)
        return self

    def teardown(self) -> Self:  # type: ignore[valid-type]
        if self._group is None:
            raise RuntimeError(f"`{type(self).__name__}` does not own a group to destroy.")
        self.destroy_group(self._group)
        self._group = None
        return self
