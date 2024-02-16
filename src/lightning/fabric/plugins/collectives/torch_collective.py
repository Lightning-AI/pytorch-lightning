import datetime
import os
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch import Tensor
from typing_extensions import Self, override

from lightning.fabric.plugins.collectives.collective import Collective
from lightning.fabric.utilities.types import CollectibleGroup, RedOpType, ReduceOp

if dist.is_available():
    from torch.distributed.constants import default_pg_timeout
else:
    default_pg_timeout = datetime.timedelta(seconds=1800)


class TorchCollective(Collective):
    """Collective operations using `torch.distributed <https://pytorch.org/docs/stable/distributed.html>`__.

    .. warning:: This is an :ref:`experimental <versioning:Experimental API>` feature which is still in development.

    """

    manages_default_group = False

    def __init__(self) -> None:
        if not dist.is_available():
            raise RuntimeError("Torch distributed is not available.")
        super().__init__()

    @property
    @override
    def group(self) -> CollectibleGroup:
        if self._group is None:
            self._group = dist.GroupMember.WORLD
        return super().group

    @property
    @override
    def rank(self) -> int:
        # local rank
        return dist.get_rank(self.group)  # type: ignore[arg-type]

    @property
    @override
    def world_size(self) -> int:
        return dist.get_world_size(self.group)  # type: ignore[arg-type]

    @override
    def broadcast(self, tensor: Tensor, src: int) -> Tensor:
        dist.broadcast(tensor, src, group=self.group)
        return tensor

    @override
    def all_reduce(self, tensor: Tensor, op: Union[str, ReduceOp, RedOpType] = "sum") -> Tensor:
        op = self._convert_to_native_op(op)
        dist.all_reduce(tensor, op=op, group=self.group)
        return tensor

    @override
    def reduce(self, tensor: Tensor, dst: int, op: Union[str, ReduceOp, RedOpType] = "sum") -> Tensor:
        op = self._convert_to_native_op(op)
        dist.reduce(tensor, dst, op=op, group=self.group)
        return tensor

    @override
    def all_gather(self, tensor_list: List[Tensor], tensor: Tensor) -> List[Tensor]:
        dist.all_gather(tensor_list, tensor, group=self.group)
        return tensor_list

    @override
    def gather(self, tensor: Tensor, gather_list: List[Tensor], dst: int = 0) -> List[Tensor]:
        dist.gather(tensor, gather_list, dst, group=self.group)
        return gather_list

    @override
    def scatter(self, tensor: Tensor, scatter_list: List[Tensor], src: int = 0) -> Tensor:
        dist.scatter(tensor, scatter_list, src, group=self.group)
        return tensor

    @override
    def reduce_scatter(
        self, output: Tensor, input_list: List[Tensor], op: Union[str, ReduceOp, RedOpType] = "sum"
    ) -> Tensor:
        op = self._convert_to_native_op(op)
        dist.reduce_scatter(output, input_list, op=op, group=self.group)
        return output

    @override
    def all_to_all(self, output_tensor_list: List[Tensor], input_tensor_list: List[Tensor]) -> List[Tensor]:
        dist.all_to_all(output_tensor_list, input_tensor_list, group=self.group)
        return output_tensor_list

    @override
    def send(self, tensor: Tensor, dst: int, tag: int = 0) -> None:
        dist.send(tensor, dst, tag=tag, group=self.group)

    @override
    def recv(self, tensor: Tensor, src: Optional[int] = None, tag: int = 0) -> Tensor:
        dist.recv(tensor, src, tag=tag, group=self.group)
        return tensor

    def all_gather_object(self, object_list: List[Any], obj: Any) -> List[Any]:
        dist.all_gather_object(object_list, obj, group=self.group)
        return object_list

    def broadcast_object_list(
        self, object_list: List[Any], src: int, device: Optional[torch.device] = None
    ) -> List[Any]:
        dist.broadcast_object_list(object_list, src, group=self.group, device=device)
        return object_list

    def gather_object(self, obj: Any, object_gather_list: List[Any], dst: int = 0) -> List[Any]:
        dist.gather_object(obj, object_gather_list, dst, group=self.group)
        return object_gather_list

    def scatter_object_list(
        self, scatter_object_output_list: List[Any], scatter_object_input_list: List[Any], src: int = 0
    ) -> List[Any]:
        dist.scatter_object_list(scatter_object_output_list, scatter_object_input_list, src, group=self.group)
        return scatter_object_output_list

    @override
    def barrier(self, device_ids: Optional[List[int]] = None) -> None:
        if self.group == dist.GroupMember.NON_GROUP_MEMBER:
            return
        dist.barrier(group=self.group, device_ids=device_ids)

    def monitored_barrier(self, timeout: Optional[datetime.timedelta] = None, wait_all_ranks: bool = False) -> None:
        dist.monitored_barrier(group=self.group, timeout=timeout, wait_all_ranks=wait_all_ranks)

    @override
    def setup(self, main_address: Optional[str] = None, main_port: Optional[str] = None, **kwargs: Any) -> Self:
        if self.is_initialized():
            return self
        # maybe set addr
        set_addr = False
        addr_key = "MASTER_ADDR"
        if main_address is not None and addr_key not in os.environ:
            os.environ[addr_key] = main_address
            set_addr = True
        # maybe set port
        set_port = False
        port_key = "MASTER_PORT"
        if main_port is not None and port_key not in os.environ:
            os.environ[port_key] = str(main_port)
            set_port = True
        # this will `init_group`
        super().setup(**kwargs)
        # set as a class attribute so any instance can know whether we initialized the default process group
        TorchCollective.manages_default_group = True
        # cleanup
        if set_addr:
            os.environ.pop("MASTER_ADDR", None)
        if set_port:
            os.environ.pop("MASTER_PORT", None)
        return self

    @override
    def teardown(self) -> Self:
        group_member = self.group != dist.GroupMember.NON_GROUP_MEMBER
        super().teardown()  # will destroy its own group
        # try to destroy the default group. this should only be done by a group member to avoid race conditions,
        # and only if the class is managing it
        if (
            group_member
            and TorchCollective.manages_default_group
            and (default_group := dist.GroupMember.WORLD) is not None  # not destroyed already
            and len(dist.distributed_c10d._pg_map) == 1  # only the default group is left
        ):
            self.destroy_group(default_group)
            TorchCollective.manages_default_group = False
        elif TorchCollective.manages_default_group and dist.GroupMember.WORLD is None:
            TorchCollective.manages_default_group = False
        return self

    @classmethod
    @override
    def is_available(cls) -> bool:
        return dist.is_available()

    @classmethod
    @override
    def is_initialized(cls) -> bool:
        return cls.is_available() and dist.is_initialized()

    @classmethod
    @override
    def init_group(cls, **kwargs: Any) -> None:
        dist.init_process_group(**kwargs)

    @classmethod
    @override
    def new_group(cls, **kwargs: Any) -> CollectibleGroup:
        return dist.new_group(**kwargs)

    @classmethod
    @override
    def destroy_group(cls, group: CollectibleGroup) -> None:
        # can be called by all processes in the default group, group will be `object()` if they are not part of the
        # current group
        if group in dist.distributed_c10d._pg_map:
            dist.destroy_process_group(group)  # type: ignore[arg-type]

    @classmethod
    @override
    def _convert_to_native_op(cls, op: Union[str, ReduceOp, RedOpType]) -> Union[ReduceOp, RedOpType]:
        # `ReduceOp` is an empty shell for `RedOpType`, the latter being the actually returned class.
        # For example, `ReduceOp.SUM` returns a `RedOpType.SUM`. the only exception is `RedOpType.PREMUL_SUM` where
        # `ReduceOp` is still the desired class, but it's created via a special `_make_nccl_premul_sum` function
        if isinstance(op, (ReduceOp, RedOpType)):
            return op
        if not isinstance(op, str):
            raise ValueError(f"Unsupported op {op!r} of type {type(op).__name__}")
        op = op.upper()
        # `ReduceOp` should contain `RedOpType`'s members
        value = getattr(ReduceOp, op, None)
        if value is None:
            raise ValueError(f"op {op!r} is not a member of `ReduceOp`")
        return value
