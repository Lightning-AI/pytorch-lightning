from typing import Any

import torch

from lightning_lite.accelerators.tpu import _XLA_AVAILABLE
from lightning_lite.plugins.collectives.collective import Collective


class XLACollective(Collective):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if not _XLA_AVAILABLE:
            raise ModuleNotFoundError(str(_XLA_AVAILABLE))
        super().__init__(*args, **kwargs)

    @staticmethod
    def _convert_to_native_op(op: str) -> str:
        import torch_xla.core.xla_model as xm

        # https://github.com/pytorch/xla/blob/28ea3758e9586ef8cc22270a16e1ddb4a21aa6f7/torch_xla/core/xla_model.py#L23-L28
        value = getattr(xm, f"REDUCE_{op.upper()}", None)
        if value is None:
            raise ValueError("TODO")
        return value

    def reduce(
        self,
        tensor: torch.Tensor,
        dst: int,
        op: str,
    ) -> torch.Tensor:
        if op != "mean":
            op = self._convert_to_native_op(op)
        if op not in ("sum", "mean"):
            raise NotImplementedError("TODO")

        import torch_xla.core.xla_model as xm

        output = xm.mesh_reduce("reduce", tensor, sum)
        if op == "mean":
            output = output / self.world_size
        return output
