import inspect
from functools import update_wrapper, partial
from typing import Any, Optional, Generator, Callable, Mapping

import torch
from torch import Tensor
from torch.optim import Optimizer
from pytorch_lightning.loops import Loop

from pytorch_lightning.loops import TrainingBatchLoop
from pytorch_lightning.utilities import AttributeDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.types import STEP_OUTPUT


class YieldLoop(TrainingBatchLoop):

    def connect(self, **kwargs: "Loop") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def on_run_start(self, *args, **kwargs):
        super().on_run_start(*args, **kwargs)
        assert inspect.isgeneratorfunction(self.trainer.lightning_module.training_step)
        assert self.trainer.lightning_module.automatic_optimization

    def advance(self, batch, batch_idx, dataloader_idx):
        split_idx, split_batch = self._remaining_splits.pop(0)
        self.batch_idx = batch_idx
        self.split_idx = split_idx

        # let logger connector extract current batch size
        self.trainer.logger_connector.on_train_split_start(batch_idx, split_idx, split_batch)

        training_step_generator = self._get_training_step_generator(split_batch, batch_idx, opt_idx=0, hiddens=self._hiddens)

        for opt_idx, optimizer in enumerate(self.trainer.optimizers):
            self.optim_progress.optimizer_idx = opt_idx
            result = self._run_optimization(training_step_generator, batch_idx, opt_idx, optimizer)
            if result:
                self.batch_outputs[opt_idx].append(result.training_step_output)

    def _run_optimization(
        self, training_step_generator: Generator, batch_idx: int, opt_idx: int, optimizer: torch.optim.Optimizer,
    ):

        # toggle model params
        self._run_optimization_start(opt_idx, optimizer)

        result = AttributeDict()
        closure = self._make_closure(training_step_generator, batch_idx, opt_idx, optimizer, result)

        if self.should_accumulate():
            # For gradient accumulation

            # -------------------
            # calculate loss (train step + train step end)
            # -------------------
            # automatic_optimization=True: perform ddp sync only when performing optimizer_step
            # automatic_optimization=False: don't block synchronization here
            with self.block_ddp_sync_behaviour():
                closure()

        # ------------------------------
        # BACKWARD PASS
        # ------------------------------
        # gradient update with accumulated gradients
        else:
            self._optimizer_step(optimizer, opt_idx, batch_idx, closure)

        if result:
            # if no result, user decided to skip optimization
            # otherwise update running loss + reset accumulated loss
            self._update_running_loss(result.loss)
            self._process_closure_result(result)

        # untoggle model params
        self._run_optimization_end(opt_idx)
        return result

    def _get_training_step_generator(self, split_batch: Any, batch_idx: int, opt_idx: int, hiddens: Tensor) -> Generator:
        step_kwargs = self._build_kwargs(split_batch, batch_idx, opt_idx, hiddens)
        generator = self.trainer.accelerator.training_step(step_kwargs)

        # self.trainer.accelerator.post_training_step()

        return generator

    def _training_step(self, training_step_generator) -> Optional[AttributeDict]:
        # give the PL module a result for logging
        model_ref = self.trainer.lightning_module

        with self.trainer.profiler.profile("model_forward"):
            # manually capture logged metrics
            model_ref._current_fx_name = "training_step"
            with self.trainer.profiler.profile("training_step"):
                training_step_output = next(training_step_generator)
                self.trainer.accelerator.post_training_step()

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            self._check_training_step_output(training_step_output)

            training_step_output = self._process_training_step_output(training_step_output)
            if training_step_output is None:
                return

        closure_loss = None
        loss = None
        if self.trainer.lightning_module.automatic_optimization:
            # accumulate loss. if accumulate_grad_batches==1, no effect
            closure_loss = training_step_output.minimize / self.trainer.accumulate_grad_batches
            # the loss will get scaled for amp. avoid any modifications to it
            loss = closure_loss.detach().clone()
        return AttributeDict(closure_loss=closure_loss, loss=loss, training_step_output=training_step_output)

    def _training_step_and_backward_closure(
        self,
        training_step_generator: Generator,
        batch_idx: int,
        opt_idx: int,
        optimizer: Optimizer,
        return_result: AttributeDict,
    ) -> Optional[Tensor]:
        result = self.training_step_and_backward(training_step_generator, batch_idx, opt_idx, optimizer)
        if result is not None:
            return_result.update(result)
            return return_result.loss

    def training_step_and_backward(
        self,
        training_step_generator: Generator,
        batch_idx: int,
        opt_idx: int,
        optimizer: torch.optim.Optimizer,
    ) -> STEP_OUTPUT:
        """Wrap forward, zero_grad and backward in a closure so second order methods work"""
        with self.trainer.profiler.profile("training_step_and_backward"):
            # lightning module hook
            result = self._training_step(training_step_generator)

            if not self._skip_backward and self.trainer.lightning_module.automatic_optimization:
                is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0

                if is_first_batch_to_accumulate:
                    self._on_before_zero_grad(optimizer)
                    self._optimizer_zero_grad(batch_idx, optimizer, opt_idx)

                # backward pass
                if result is not None:
                    with self.trainer.profiler.profile("backward"):
                        self.backward(result, optimizer, opt_idx)

                    # when in dev debugging track the losses
                    self.trainer.dev_debugger.track_train_loss_history(batch_idx, result.loss)

                    # check if loss or model weights are nan
                    if self.trainer.terminate_on_nan:
                        self._check_finite(result.loss)

                else:
                    self._warning_cache.warn(
                        "training_step returned None. If this was on purpose, ignore this warning..."
                    )

        return result
