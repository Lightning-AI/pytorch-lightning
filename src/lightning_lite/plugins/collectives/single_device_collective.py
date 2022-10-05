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
        *_: Any,
        **__: Any,
    ) -> List[torch.Tensor]:
        return tensor_list

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        *_: Any,
        **__: Any,
    ) -> Optional[List[torch.Tensor]]:
        return gather_list

    def scatter(
        self,
        tensor: torch.Tensor,
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return tensor

    def reduce_scatter(
        self,
        output: torch.Tensor,
        *_: Any,
        **__: Any,
    ) -> torch.Tensor:
        return output

    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        *_: Any,
        **__: Any,
    ) -> List[torch.Tensor]:
        return output_tensor_list

    def barrier(
        self,
        *_: Any,
        **__: Any,
    ) -> None:
        pass
