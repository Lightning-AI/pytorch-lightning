import inspect
from functools import partial
from typing import Any, Generator, Optional, List

from torch import Tensor
from torch.optim import Optimizer

from pytorch_lightning.loops import Loop, OptimizerLoop
from pytorch_lightning.loops.utilities import (
    _check_training_step_output,
    _process_training_step_output,
    _build_training_step_kwargs,
)
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class Yield:
    """
    Interface for the LightningModule to define a flavor for automatic optimization where
    the training step method yields losses for each optimizer instead of returning them.
    """

    def training_step(self, batch, batch_idx, optimizer_idx=0) -> Generator:
        # the optimizer_idx is just here to shortcut the implementation for this POC
        # TODO: generalize and override the build_kwargs function in YieldLoop
        pass


class YieldLoop(OptimizerLoop):
    def __init__(self):
        super().__init__()
        self._training_step_generator: Generator = ...

    def connect(self, **kwargs: "Loop") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def on_run_start(self, batch: Any, hiddens: Any, optimizers: List[Optimizer], batch_idx: int):
        super().on_run_start(batch, hiddens, optimizers, batch_idx)
        if not isinstance(self.trainer.lightning_module, Yield):
            raise MisconfigurationException(
                "Given LightingModule does not inherit the Yield interface for automatic optimization, but a"
                " YieldLoop was requested."
            )
        assert inspect.isgeneratorfunction(self.trainer.lightning_module.training_step)
        assert self.trainer.lightning_module.automatic_optimization

        self._training_step_generator = self._get_training_step_generator(batch, batch_idx, opt_idx=0, hiddens=hiddens)

    def _make_step_fn(self, split_batch, batch_idx, opt_idx, hiddens):
        return partial(self._training_step, self._training_step_generator)

    def _get_training_step_generator(
        self, split_batch: Any, batch_idx: int, opt_idx: int, hiddens: Tensor
    ) -> Generator:
        step_kwargs = _build_training_step_kwargs(
            self.trainer.lightning_module, self.trainer.optimizers, split_batch, batch_idx, opt_idx, hiddens
        )
        generator = self.trainer.accelerator.training_step(step_kwargs)
        return generator

    def _training_step(self, training_step_generator: Generator) -> Optional[AttributeDict]:
        model_ref = self.trainer.lightning_module

        with self.trainer.profiler.profile("model_forward"):

            # manually capture logged metrics
            model_ref._current_fx_name = "training_step"
            with self.trainer.profiler.profile("training_step"):
                training_step_output = next(training_step_generator)
                self.trainer.accelerator.post_training_step()

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            _check_training_step_output(self.trainer.lightning_module, training_step_output)

            result_collection, self._hiddens = _process_training_step_output(self.trainer, training_step_output)
            if result_collection is None:
                return

        # accumulate loss. if accumulate_grad_batches==1, no effect
        closure_loss = result_collection.minimize / self.trainer.accumulate_grad_batches
        # the loss will get scaled for amp. avoid any modifications to it
        loss = closure_loss.detach().clone()
        return AttributeDict(closure_loss=closure_loss, loss=loss, result_collection=result_collection)
