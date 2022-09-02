import pytorch_lightning as pl
from typing import Optional, Any, Union
from torch import Tensor
from torch.optim import Optimizer
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.warnings import WarningCache


warning_cache = WarningCache()


class ColossalAIPrecisionPlugin(PrecisionPlugin):
    def __init__(self) -> None:
        super().__init__()
        self.precision = 16

    def backward(self, model: "pl.LightningModule", closure_loss: Tensor, optimizer: Optional[Optimizer], optimizer_idx: Optional[int], *args: Any, **kwargs: Any) -> None:
        if is_overridden("backward", model):
            warning_cache.warn(
                "You have overridden the `LightningModule.backward` hook but it will be ignored since ColossalAI handles"
                " the backward logic internally."
            )
        return optimizer.backward(closure_loss)

    def clip_grad_by_norm(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        optimizer.clip_grad_norm(None, clip_val)

    def clip_grad_by_value(self, optimizer: Optimizer, clip_val: Union[int, float]) -> None:
        raise MisconfigurationException('clip_grad_by_value is not supported by `Colossalai`')

    def optimizer_step(self, model, optimizer, optimizer_idx: int, closure, **kwargs: Any) -> Any:
        closure_result = closure()
        self._after_closure(model, optimizer, optimizer_idx)
        skipped_backward = closure_result is None
        if isinstance(model, pl.LightningModule) and model.automatic_optimization and skipped_backward:
            raise MisconfigurationException(
                "Skipping backward by returning `None` from your `training_step` is not supported by `Colossalai`"
            )
        optimizer.step()

    def _track_grad_norm(self, trainer: "pl.Trainer") -> None:
        if trainer.track_grad_norm == -1:
            return
        # the gradients are not available in the model due to gradient partitioning in zero stage >= 2
        warning_cache.warn(
            f"You set `Trainer(track_grad_norm={trainer.track_grad_norm!r})' but this is not supported for ColossalAI."
            " The setting will be ignored."
        )
