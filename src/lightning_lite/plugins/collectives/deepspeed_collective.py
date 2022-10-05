import datetime
from typing import Any, List, Optional

import torch

from lightning_lite.plugins.collectives.collective import Collective
from lightning_lite.strategies.deepspeed import _DEEPSPEED_AVAILABLE
from lightning_lite.utilities.types import CollectibleGroup

if _DEEPSPEED_AVAILABLE:
    import deepspeed.comm as dist


class DeepSpeedCollective(Collective):
    def __init__(self, instantiate_group: bool = False, **group_kwargs: Any) -> None:
        if not _DEEPSPEED_AVAILABLE:
            raise RuntimeError("Torch distributed is not available.")
        super().__init__(instantiate_group, **group_kwargs)

    @property
    def rank(self) -> int:
        return dist.get_rank(self.group)

    @property
    def world_size(self) -> int:
        return dist.get_world_size(self.group)

    @staticmethod
    def init_group(
        **kwargs: Any,
    ) -> CollectibleGroup:
        return dist.init_process_group(**kwargs)

    @staticmethod
    def destroy_group(group: CollectibleGroup) -> None:
        dist.destroy_process_group(group)

    @staticmethod
    def _convert_to_native_op(op: str) -> "dist.ReduceOp":
        value = getattr(dist.ReduceOp, op.upper(), None)
        if value is None:
            raise ValueError("TODO")
        return value

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
    ) -> torch.Tensor:
        dist.broadcast(tensor, src, group=self.group)
        return tensor

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: str = "sum",
    ) -> torch.Tensor:
        op = self._convert_to_native_op(op)
        dist.all_reduce(tensor, op=op, group=self.group)
        return tensor

    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
        op: str = "sum",
    ) -> torch.Tensor:
        op = self._convert_to_native_op(op)
        dist.reduce(tensor, dst, op=op, group=self.group)
        return tensor

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        dist.all_gather(tensor_list, tensor, group=self.group)
        return tensor_list

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        dst: int = 0,
    ) -> Optional[List[torch.Tensor]]:
        dist.gather(tensor, gather_list, dst, group=self.group)
        return gather_list

    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
    ) -> torch.Tensor:
        dist.scatter(tensor, scatter_list, src, group=self.group)
        return tensor

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: str = "sum",
    ) -> torch.Tensor:
        op = self._convert_to_native_op(op)
        dist.reduce_scatter(output, input_list, op=op, group=self.group)
        return output

    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        dist.all_to_all(output_tensor_list, input_tensor_list, group=self.group)
        return output_tensor_list

    def barrier(
        self,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        dist.barrier(group=self.group, device_ids=device_ids)

    def all_gather_object(
        self,
        object_list: List[Any],
        object: Any,
    ) -> List[Any]:
        dist.all_gather_object(object_list, object, group=self.group)
        return object_list

    def broadcast_object_list(
        self,
        object_list: List[Any],
        src: int,
        device: Optional[torch.device] = None,
    ) -> List[Any]:
        dist.broadcast_object_list(object_list, src, group=self.group, device=device)
        return object_list

    def gather_object(
        self,
        obj: Any,
        object_gather_list: Optional[List[Any]] = None,
        dst: int = 0,
    ) -> Optional[List[Any]]:
        dist.gather_object(obj, object_gather_list, dst, group=self.group)
        return object_gather_list

    def scatter_object_list(
        self,
        scatter_object_output_list: List[Any],
        scatter_object_input_list: Optional[List[Any]],
        src: int = 0,
    ) -> List[Any]:
        dist.scatter_object_list(scatter_object_output_list, scatter_object_input_list, src, group=self.group)
        return scatter_object_output_list

    def monitored_barrier(
        self,
        timeout: Optional[datetime.timedelta] = None,
        wait_all_ranks: bool = False,
    ) -> None:
        dist.monitored_barrier(group=self.group, timeout=timeout, wait_all_ranks=wait_all_ranks)
