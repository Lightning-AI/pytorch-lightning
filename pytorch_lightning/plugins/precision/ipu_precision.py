from typing import Any

from torch import Tensor

from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin


class IPUPrecisionPlugin(PrecisionPlugin):

    def __init__(self, precision: int) -> None:
        super().__init__()
        self.precision = precision

    def backward(
        self,
        closure_loss: Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        # IPU internally manages bwd step.
        return closure_loss

    def clip_gradients(self, *args, **kwargs) -> None:
        pass
