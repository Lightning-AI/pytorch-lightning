from typing import Any, List, Optional

import torch

from lightning_lite.plugins.collectives.collective import Collective
from lightning_lite.utilities.types import ProcessGroup


class SingleDeviceCollective(Collective):
    @property
    def rank(self) -> int:
        return 0

    @property
    def world_size(self) -> int:
        return 1

    @staticmethod
    def init_group(
        **kwargs: Any,
    ) -> ProcessGroup:
        return object()  # type: ignore[return-value]

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

    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: List[torch.Tensor],
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

    def barrier(
        self,
        *_: Any,
        **__: Any,
    ) -> None:
        pass
