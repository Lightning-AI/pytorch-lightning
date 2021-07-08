from collections import OrderedDict
from typing import Any, Dict, Iterator, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.optim import Optimizer

from pytorch_lightning.loops import Loop
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection


class SimpleLoop(Loop):
    """
    This loop is for demonstration purposes only.
    It implements a purely iteration-based loop with a bare miminum of functionality.
    - 1 optimizer
    - no logging
    - no grad accumulation
    - no epoch hooks calling
    """

    def __init__(self, num_iterations: int = float("inf")):
        super().__init__()
        self.num_iterations = num_iterations
        self.train_dataloader: Optional[Iterator] = None

        # required for trainer and logger connector
        self._results = ResultCollection(training=True)

    @property
    def global_step(self) -> int:
        return self.iteration_count

    @property
    def batch_idx(self) -> int:
        # required by progress bar
        return self.iteration_count

    @property
    def running_loss(self) -> Tensor:
        # required by progress bar
        return torch.tensor(123.)

    @property
    def current_epoch(self) -> int:
        return 0

    @property
    def skip(self) -> bool:
        return self.done or self.trainer.num_training_batches == 0

    @property
    def done(self) -> bool:
        return self.iteration_count >= self.num_iterations

    def reset(self) -> None:
        self.iteration_count = 0

    def on_run_start(self) -> None:
        self.train_dataloader = iter(self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader))
        self.trainer.call_hook("on_train_start")

    def advance(self) -> None:
        batch = next(self.train_dataloader)

        opt_idx = 0
        optimizer = self.trainer.optimizers[opt_idx]

        self.trainer.call_hook("on_train_batch_start", batch, self.iteration_count, dataloader_idx=0)

        output = self._run_optimization(batch, self.iteration_count, optimizer)

        # hook
        self.trainer.call_hook("on_train_batch_end", output, batch, self.iteration_count, dataloader_idx=0)
        self.trainer.call_hook("on_batch_end")

    def on_run_end(self) -> None:
        self.trainer.call_hook("on_train_end")
        self.trainer.accelerator.on_train_end()
        self.trainer._running_stage = None

    def _run_optimization(self, batch: Any, batch_idx: int, optimizer: Optimizer):
        lightning_module = self.trainer.lightning_module

        # lightning module training_step
        step_kwargs = OrderedDict([("batch", batch), ("batch_idx", batch_idx)])
        lightning_module._current_fx_name = "training_step"
        training_step_output = self.trainer.accelerator.training_step(step_kwargs)
        self.trainer.accelerator.post_training_step()

        training_step_output = self.trainer.call_hook("training_step_end", training_step_output)
        loss, extra = self._process_training_step_output(training_step_output)

        # backward pass (single optimizer, no accumulation supported)
        self.trainer.accelerator.backward(loss, optimizer, optimizer_idx=0, should_accumulate=False)

        # optimizer step (no closures supported)
        lightning_module.optimizer_step(optimizer=optimizer)

        output = extra
        output["loss"] = loss.detach()
        return output

    @staticmethod
    def _process_training_step_output(training_step_output: Union[Dict, Tensor]) -> Tuple[Tensor, Dict]:
        loss = None
        extra = {}

        if isinstance(training_step_output, dict):
            loss = training_step_output.pop("loss")
            extra = training_step_output

        elif isinstance(training_step_output, Tensor):
            loss = training_step_output

        return loss, extra
