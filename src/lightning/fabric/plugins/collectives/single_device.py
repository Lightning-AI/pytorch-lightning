from typing import Any

from torch import Tensor
from typing_extensions import override

from lightning.fabric.plugins.collectives.collective import Collective
from lightning.fabric.utilities.types import CollectibleGroup


class SingleDeviceCollective(Collective):
    """Support for collective operations on a single device (no-op).

    .. warning:: This is an :ref:`experimental <versioning:Experimental API>` feature which is still in development.

    """

    @property
    @override
    def rank(self) -> int:
        return 0

    @property
    @override
    def world_size(self) -> int:
        return 1

    @override
    def broadcast(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
        return tensor

    @override
    def all_reduce(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
        return tensor

    @override
    def reduce(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
        return tensor

    @override
    def all_gather(self, tensor_list: list[Tensor], tensor: Tensor, **__: Any) -> list[Tensor]:
        return [tensor]

    @override
    def gather(self, tensor: Tensor, *_: Any, **__: Any) -> list[Tensor]:
        return [tensor]

    @override
    def scatter(
        self,
        tensor: Tensor,
        scatter_list: list[Tensor],
        *_: Any,
        **__: Any,
    ) -> Tensor:
        return scatter_list[0]

    @override
    def reduce_scatter(self, output: Tensor, input_list: list[Tensor], *_: Any, **__: Any) -> Tensor:
        return input_list[0]

    @override
    def all_to_all(
        self, output_tensor_list: list[Tensor], input_tensor_list: list[Tensor], *_: Any, **__: Any
    ) -> list[Tensor]:
        return input_tensor_list

    @override
    def send(self, *_: Any, **__: Any) -> None:
        pass

    @override
    def recv(self, tensor: Tensor, *_: Any, **__: Any) -> Tensor:
        return tensor

    @override
    def barrier(self, *_: Any, **__: Any) -> None:
        pass

    @classmethod
    @override
    def is_available(cls) -> bool:
        return True  # vacuous truth

    @classmethod
    @override
    def is_initialized(cls) -> bool:
        return True  # vacuous truth

    @classmethod
    @override
    def init_group(cls, **_: Any) -> None:
        pass

    @classmethod
    @override
    def new_group(cls, **_: Any) -> CollectibleGroup:
        return object()  # type: ignore[return-value]

    @classmethod
    @override
    def destroy_group(cls, group: CollectibleGroup) -> None:
        pass

    @classmethod
    @override
    def _convert_to_native_op(cls, op: str) -> str:
        return op
