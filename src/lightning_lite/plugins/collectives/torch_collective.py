import datetime
from typing import Any, List, Optional

import torch
from torch.distributed import ReduceOp

from lightning_lite.plugins.collectives.collective import Collective
from lightning_lite.utilities.types import ProcessGroup


class TorchCollective(Collective):
    @property
    def rank(self) -> int:
        return torch.distributed.get_rank(self.group)

    @property
    def world_size(self) -> int:
        return torch.distributed.get_world_size(self.group)

    @staticmethod
    def init_group(
        **kwargs: Any,
    ) -> ProcessGroup:
        return torch.distributed.init_process_group(**kwargs)

    def broadcast(
        self,
        tensor: torch.Tensor,
        src: int,
    ) -> torch.Tensor:
        torch.distirbuted.broadcast(tensor, src, group=self.group)
        return tensor

    def broadcast_object_list(
        self,
        object_list: List[Any],
        src: int,
        device: Optional[torch.device] = None,
    ) -> List[Any]:
        torch.distributed.broadcast_object_list(object_list, src, group=self.group, device=device)

    def all_reduce(
        self,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
    ) -> torch.Tensor:
        torch.distributed.all_reduce(tensor, op=op, group=self.group)
        return tensor

    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
        op: ReduceOp = ReduceOp.SUM,
    ) -> torch.Tensor:
        torch.distributed.reduce(tensor, dst, op=op, group=self.group)
        return tensor

    def all_gather(
        self,
        tensor_list: List[torch.Tensor],
        tensor: torch.Tensor,
    ) -> List[torch.Tensor]:
        torch.distributed.all_gather(tensor_list, tensor, group=self.group)
        return tensor_list

    def all_gather_object(
        self,
        object_list: List[Any],
        object: Any,
    ) -> List[Any]:
        torch.distributed.all_gather_object(object_list, object, group=self.group)
        return object_list

    def gather(
        self,
        tensor: torch.Tensor,
        gather_list: Optional[List[torch.Tensor]] = None,
        dst: int = 0,
    ) -> Optional[List[torch.Tensor]]:
        torch.distributed.gather(tensor, gather_list, dst, group=self.group)
        return gather_list

    def gather_object(
        self,
        obj: Any,
        object_gather_list: Optional[List[Any]] = None,
        dst: int = 0,
    ) -> Optional[List[Any]]:
        torch.distributed.gather_object(obj, object_gather_list, dst, group=self.group)
        return object_gather_list

    def scatter(
        self,
        tensor: torch.Tensor,
        scatter_list: Optional[List[torch.Tensor]] = None,
        src: int = 0,
        async_op: bool = False,
    ) -> torch.Tensor:
        torch.distributed.scatter(tensor, scatter_list, src, group=self.group, async_op=async_op)
        return tensor

    def scatter_object_list(
        self,
        scatter_object_output_list: List[Any],
        scatter_object_input_list: Optional[List[Any]],
        src: int = 0,
    ) -> List[Any]:
        torch.distributed.scatter_object_list(
            scatter_object_output_list, scatter_object_input_list, src, group=self.group
        )
        return scatter_object_output_list

    def reduce_scatter(
        self,
        output: torch.Tensor,
        input_list: List[torch.Tensor],
        op: ReduceOp = ReduceOp.SUM,
    ) -> torch.Tensor:
        torch.distributed.reduce_scatter(output, input_list, op=op, group=self.group)
        return output

    def all_to_all(
        self,
        output_tensor_list: List[torch.Tensor],
        input_tensor_list: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        torch.distributed.all_to_all(output_tensor_list, input_tensor_list, group=self.group)
        return output_tensor_list

    def barrier(
        self,
        device_ids: Optional[List[int]] = None,
    ) -> None:
        torch.distributed.barrier(group=self.group, device_ids=device_ids)

    def monitored_barrier(
        self,
        timeout: Optional[datetime.timedelta] = None,
        wait_all_ranks: bool = False,
    ) -> None:
        torch.distributed.monitored_barrier(group=self.group, timeout=timeout, wait_all_ranks=wait_all_ranks)
