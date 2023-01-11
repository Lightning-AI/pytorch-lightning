from typing import Any, List

from torch import Tensor

from lightning_fabric.plugins.collectives.collective import Collective
from lightning_fabric.utilities.types import CollectibleGroup


class SingleDeviceCollective(Collective):
    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    def broadcast(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
        return tensor

    def all_reduce(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
        return tensor

    def reduce(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
        return tensor

    def all_gather(self, tensor_list: List[Tensor], tensor: Tensor, **__: Any) -> List[Tensor]:
        return [tensor]

    def gather(self, tensor: Tensor, *_: Any, **__: Any) -> List[Tensor]:
        return [tensor]

    def scatter(
        self,
        tensor: Tensor,
        scatter_list: List[Tensor],
        *_: Any,
        **__: Any,
    ) -> Tensor:
        return scatter_list[0]

    def reduce_scatter(self, output: Tensor, input_list: List[Tensor], *_: Any, **__: Any) -> Tensor:
        return input_list[0]

    def all_to_all(
        self, output_tensor_list: List[Tensor], input_tensor_list: List[Tensor], *_: Any, **__: Any
    ) -> List[Tensor]:
        return input_tensor_list

    def send(self, *_: Any, **__: Any) -> None:
        pass

    def recv(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
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
