import pytorch_lightning as pl
from typing import Optional, Any, Union
from torch import Tensor
from torch.optim import Optimizer
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin


class ColossalAIPrecisionPlugin(PrecisionPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.precision = 16

    def backward(self, model: "pl.LightningModule", closure_loss: Tensor, optimizer: Optional[Optimizer], optimizer_idx: Optional[int], *args: Any, **kwargs: Any) -> None:
        return optimizer.backward(closure_loss)

    def clip_grad_by_norm(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        optimizer.clip_grad_norm(None, clip_val)

    def optimizer_step(self, model, optimizer, optimizer_idx: int, closure, **kwargs: Any) -> Any:
        closure_result = closure()
        self._after_closure(model, optimizer, optimizer_idx)
        optimizer.step()
