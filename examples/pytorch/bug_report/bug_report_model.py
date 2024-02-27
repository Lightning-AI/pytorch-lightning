from contextlib import contextmanager
from typing import Generator, Literal, Optional

import torch
from lightning.pytorch.plugins.precision import MixedPrecisionPlugin


class PipelineMixedPrecisionPlugin(MixedPrecisionPlugin):
    """Overrides PTL autocasting to not wrap training/val/test_step. We do this because we have the megatron-core
    fwd/bwd functions in training_step. This means .backward is being called in training_step so we do not want the
    whole step wrapped in autocast.

    We instead wrap the fwd_output_and_loss_func that is passed to the megatron-core fwd/bwd functions.

    """

    def __init__(
        self,
        precision: Literal["16-mixed", "bf16-mixed"],
        device: str,
        scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ) -> None:
        super().__init__(precision, device, scaler=scaler)
        dtype = None
        # MixedPrecisionPlugin class in PTL >= 2.0 takes only "16-mixed" or "bf16-mixed" for precision arg
        if precision == "16-mixed":
            dtype = torch.float16
        elif precision == "bf16-mixed":
            dtype = torch.bfloat16

        torch.set_autocast_gpu_dtype(dtype)

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Have the PTL context manager do nothing."""
        yield


PipelineMixedPrecisionPlugin(precision="16-mixed", device="cuda:0")
