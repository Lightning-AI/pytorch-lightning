from typing import Any, List

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

    def broadcast(self, tensor: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        return tensor

    def all_reduce(self, tensor: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        return tensor

    def reduce(self, tensor: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        return tensor

    def all_gather(self, tensor_list: List[torch.Tensor], tensor: torch.Tensor, **__: Any) -> List[torch.Tensor]:
        return [tensor]

    def gather(self, tensor: torch.Tensor, *_: Any, **__: Any) -> List[torch.Tensor]:
        return [tensor]

    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: List[torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return scatter_list[0]

    def reduce_scatter(self, output: torch.Tensor, input_list: List[torch.Tensor], *_: Any, **__: Any) -> torch.Tensor:
        return input_list[0]

    def all_to_all(
        self, output_tensor_list: List[torch.Tensor], input_tensor_list: List[torch.Tensor], *_: Any, **__: Any
    ) -> List[torch.Tensor]:
        return input_tensor_list

    def send(self, *_: Any, **__: Any) -> None:
        pass

    def recv(self, tensor: torch.Tensor, *_: Any, **__: Any) -> torch.Tensor:
        return tensor

    def barrier(self, *_: Any, **__: Any) -> None:
        pass

    @classmethod
    def is_available(cls) -> bool:
        return True  # vacuous truth

    @classmethod
    def is_initialized(cls) -> bool:
        return True  # vacuous truth

    @classmethod
    def init_group(cls, **_: Any) -> None:
        pass

    @classmethod
    def new_group(cls, **_: Any) -> CollectibleGroup:
        return object()  # type: ignore[return-value]

    @classmethod
    def destroy_group(cls, group: CollectibleGroup) -> None:
        pass

    @classmethod
    def _convert_to_native_op(cls, op: str) -> str:
        return op
