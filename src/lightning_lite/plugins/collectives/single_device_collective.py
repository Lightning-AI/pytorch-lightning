from typing import Any, List, Optional

import torch

from lightning_lite.plugins.collectives.collective import Collective
from lightning_lite.utilities.types import CollectibleGroup


class SingleDeviceCollective(Collective):
    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def broadcast(
        self,
        tensor: torch.Tensor,
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return tensor

    def all_reduce(
        self,
        tensor: torch.Tensor,
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return tensor

    def reduce(
        self,
        tensor: torch.Tensor,
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return tensor

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
        **__: Any,
    ) -> List[torch.Tensor]:
        return [tensor]

    def gather(
        self,
        tensor: torch.Tensor,
        *_: Any,
        **__: Any,
    ) -> Optional[List[torch.Tensor]]:
        return [tensor]

    def scatter(  # type: ignore[override]
        self,
        tensor: torch.Tensor,
        scatter_list: List[torch.Tensor],  # it doesn't make sense to have a None here for a single device
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return scatter_list[0]

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return input_list[0]

    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> List[torch.Tensor]:
        return input_tensor_list

    def send(self, *_: Any, **__: Any) -> None:
        pass

    def recv(self, tensor: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        return tensor

    def barrier(
        self,
        *_: Any,
        **__: Any,
    ) -> None:
        pass

    @staticmethod
    def init_group(
        **kwargs: Any,
    ) -> CollectibleGroup:
        return object()  # type: ignore[return-value]

    @staticmethod
    def destroy_group(group: CollectibleGroup) -> None:
        pass

    @staticmethod
    def _convert_to_native_op(op: str) -> str:
        return op
