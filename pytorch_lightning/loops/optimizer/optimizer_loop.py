# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer

from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.closure import Closure, ClosureResult
from pytorch_lightning.loops.utilities import (
    _block_parallel_sync_behavior,
    _build_training_step_kwargs,
    _check_training_step_output,
    _process_training_step_output,
)
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.progress import OptimizationProgress
from pytorch_lightning.utilities import AMPType, AttributeDict, DeviceType, grad_norm
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
from pytorch_lightning.utilities.imports import _TPU_AVAILABLE

_OUTPUTS_TYPE = List[List[Optional[ResultCollection]]]


class OptimizerLoop(Loop):
    """Runs over a sequence of optimizers. This loop implements what is known in Lightning as Automatic Optimization."""

    def __init__(self):
        super().__init__()
        # TODO: use default dict here to simplify logic in loop
        self.outputs: _OUTPUTS_TYPE = []
        self.optim_progress: OptimizationProgress = OptimizationProgress()

        self._skip_backward: bool = False
        self._batch_idx: int = 0
        self._optimizers: List[Optimizer] = []
        self._hiddens: Optional[Any] = None

    @property
    def done(self) -> bool:
        """Returns ``True`` when the last optimizer in the sequence has run."""
        return self.optim_progress.optimizer_idx >= len(self._optimizers)

    def connect(self, **kwargs: "Loop") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def reset(self) -> None:
        if not self.restarting:
            self.optim_progress.optimizer_idx = 0
        self.outputs = [[] for _ in range(len(self.trainer.optimizers))]

    def on_run_start(self, batch: Any, hiddens: Any, optimizers: List[Optimizer], batch_idx: int) -> None:  # type: ignore[override]
        self._batch_idx = batch_idx
        self._optimizers = optimizers

    def advance(self, batch: Any, hiddens: Any, *args, **kwargs) -> None:  # type: ignore[override]
        self._hiddens = hiddens
        result = self._run_optimization(
            batch,
            self._batch_idx,
            self._optimizers[self.optim_progress.optimizer_idx],
            self.optim_progress.optimizer_idx,
        )
        if result:
            self.outputs[self.optim_progress.optimizer_idx].append(deepcopy(result.result_collection))

        self.optim_progress.optimizer_idx += 1

    def on_run_end(self) -> Tuple[_OUTPUTS_TYPE, Optional[Any]]:
        outputs = self.outputs
        hiddens = self._hiddens
        # free memory
        self.outputs = []
        self._hiddens = None
        return outputs, hiddens

    def backward(
        self,
        loss: Tensor,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """Performs the backward step.

        Args:
            loss: The loss value to back-propagate on
            optimizer: Current optimizer being used
            opt_idx: Index of the current optimizer being used
        """
        self.trainer.accelerator.backward(loss, optimizer, opt_idx, *args, **kwargs)

        if not self.trainer.fit_loop.should_accumulate():
            # track gradients
            grad_norm_dict = self._track_and_norm_grad(optimizer=optimizer)
            if grad_norm_dict:
                self.trainer.lightning_module._current_fx_name = "on_after_backward"
                self.trainer.lightning_module.log_grad_norm(grad_norm_dict)
        return loss

    def _run_optimization(
        self,
        split_batch: Any,
        batch_idx: int,
        optimizer: torch.optim.Optimizer,
        opt_idx: int,
    ) -> Optional[ClosureResult]:
        """Runs closure (train step + backward) together with optimization if necessary.

        Args:
            split_batch: the current tbptt split of the whole batch
            batch_idx: the index of the current batch
            optimizer: the current optimizer
            opt_idx: the index of the current optimizer
        """
        # toggle model params
        self._run_optimization_start(opt_idx, optimizer)

        closure = self._make_closure(split_batch, batch_idx, opt_idx, optimizer, self._hiddens)

        if self.trainer.fit_loop.should_accumulate():
            # For gradient accumulation

            # -------------------
            # calculate loss (train step + train step end)
            # -------------------
            # automatic_optimization=True: perform ddp sync only when performing optimizer_step
            with _block_parallel_sync_behavior(self.trainer, block=True):
                closure()

        # ------------------------------
        # BACKWARD PASS
        # ------------------------------
        # gradient update with accumulated gradients
        else:
            self._optimizer_step(optimizer, opt_idx, batch_idx, closure)

        result = closure.get_result()

        if result:
            # if no result, user decided to skip optimization
            # otherwise update running loss + reset accumulated loss
            # TODO: find proper way to handle updating running loss
            assert self.trainer.fit_loop is not None
            assert self.trainer.fit_loop.epoch_loop is not None
            assert self.trainer.fit_loop.epoch_loop.batch_loop is not None
            assert result.loss is not None
            self.trainer.fit_loop.epoch_loop.batch_loop._update_running_loss(result.loss)

        # untoggle model params
        self._run_optimization_end(opt_idx)
        return result

    def _make_closure(
        self,
        split_batch: Any,
        batch_idx: int,
        opt_idx: int,
        optimizer: Optimizer,
        hiddens: Any,
    ) -> Closure:
        """
        Build a closure object that captures the given arguments and runs the `training_step` function and optionally
        other functions such as `backward` and `zero_grad`.
        """
        step_fn = self._make_step_fn(split_batch, batch_idx, opt_idx, hiddens)
        backward_fn = self._make_backward_fn(optimizer, opt_idx)
        zero_grad_fn = self._make_zero_grad_fn(batch_idx, opt_idx, optimizer)

        return Closure(
            step_fn=step_fn,
            backward_fn=backward_fn,
            zero_grad_fn=zero_grad_fn,
            profiler=self.trainer.profiler,
        )

    def _make_step_fn(
        self, split_batch: Any, batch_idx: int, opt_idx: int, hiddens: Any
    ) -> Callable[[], Optional[AttributeDict]]:
        """Build the step function that runs the `training_step` and processes its output."""
        return partial(self._training_step, split_batch, batch_idx, opt_idx, hiddens)

    def _make_zero_grad_fn(self, batch_idx: int, opt_idx: int, optimizer: Optimizer) -> Optional[Callable[[], None]]:
        """
        Build a `zero_grad` function that zeroes the gradients before back-propagation.
        Returns ``None`` in the case backward needs to be skipped.
        """

        def zero_grad_fn():
            self._on_before_zero_grad(optimizer)
            self._optimizer_zero_grad(batch_idx, optimizer, opt_idx)

        is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0
        if not self._skip_backward and is_first_batch_to_accumulate:
            return zero_grad_fn

        return None

    def _make_backward_fn(
        self,
        optimizer: Optimizer,
        opt_idx: int,
    ) -> Optional[Callable[[Tensor], Tensor]]:
        """
        Build a `backward` function that handles back-propagation through the output produced by the `training_step`
        function. Returns ``None`` in the case backward needs to be skipped.
        """

        def backward_fn(loss: Tensor):
            self.backward(loss, optimizer, opt_idx)

            # check if model weights are nan
            if self.trainer.terminate_on_nan:
                detect_nan_parameters(self.trainer.lightning_module)

            return loss

        if not self._skip_backward:
            return backward_fn

        return None

    def _run_optimization_start(self, opt_idx: int, optimizer: torch.optim.Optimizer) -> None:
        """Toggles the optimizer to ensure the correct one is used and prevend dangling grads.

        Args:
            opt_idx: the index of the optimizer to use
            optimizer: the optimizer to use

        """
        # make sure only the gradients of the current optimizer's parameters are calculated
        # in the training step to prevent dangling gradients in multiple-optimizer setup.
        if len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.toggle_optimizer(optimizer, opt_idx)

    def _run_optimization_end(self, opt_idx: int) -> None:
        if len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.untoggle_optimizer(opt_idx)

    def _optimizer_step(
        self, optimizer: torch.optim.Optimizer, opt_idx: int, batch_idx: int, train_step_and_backward_closure: Callable
    ) -> None:
        """Performs the optimizer step and some sanity checking.

        Args:
            optimizer: the optimizer to perform the step with
            opt_idx: the index of the current :param:`optimizer`
            batch_idx: the index of the current batch
            train_step_and_backward_closure: the closure function performing the train step and computing the
                gradients. By default called by the optimizer (if possible)
        """
        model_ref = self.trainer.lightning_module

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        using_native_amp = self.trainer.amp_backend is not None and self.trainer.amp_backend == AMPType.NATIVE

        # native amp + lbfgs is a no go right now
        if using_native_amp and is_lbfgs:
            raise MisconfigurationException(
                "native PyTorch amp and lbfgs are not compatible."
                " To request, please file a Github issue in PyTorch and tag @mcarilli"
            )

        # wraps into LightningOptimizer only for running step
        optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer, opt_idx)

        self.optim_progress.optimizer.step.increment_ready()

        # model hook
        model_ref.optimizer_step(
            self.trainer.current_epoch,
            batch_idx,
            optimizer,
            opt_idx,
            train_step_and_backward_closure,
            on_tpu=(self.trainer._device_type == DeviceType.TPU and _TPU_AVAILABLE),
            using_native_amp=using_native_amp,
            using_lbfgs=is_lbfgs,
        )

        self.optim_progress.optimizer.step.increment_completed()

    def _on_before_zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Calls the ``on_before_zero_grad`` hook.

        Args:
            optimizer: the current optimizer
        """
        self.optim_progress.optimizer.zero_grad.increment_ready()
        self.trainer.call_hook("on_before_zero_grad", optimizer)
        self.optim_progress.optimizer.zero_grad.increment_started()

    def _optimizer_zero_grad(self, batch_idx: int, optimizer: torch.optim.Optimizer, opt_idx: int) -> None:
        """Zeroes out all gradients of parameters optimized by the current optimizer.

        Args:
            batch_idx: the index of the current batch
            optimizer: the current optimizer
            opt_idx: the index of the current optimizer
        """
        self.trainer.accelerator.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)
        self.optim_progress.optimizer.zero_grad.increment_completed()

    def _training_step(
        self, split_batch: Any, batch_idx: int, opt_idx: int, hiddens: Tensor
    ) -> Optional[AttributeDict]:
        """Performs the actual train step with the tied hooks.

        Args:
            split_batch: the current tbptt split of the current batch
            batch_idx: the index of the current batch
            opt_idx: the index of the current optimizer
            hiddens: the model's hidden state of the previous iteration

        Returns:
            an AttributeDict containing the loss value and the training step output.
        """
        # give the PL module a result for logging
        model_ref = self.trainer.lightning_module

        with self.trainer.profiler.profile("model_forward"):

            step_kwargs = _build_training_step_kwargs(
                self.trainer.lightning_module, self.trainer.optimizers, split_batch, batch_idx, opt_idx, hiddens
            )

            # manually capture logged metrics
            model_ref._current_fx_name = "training_step"
            with self.trainer.profiler.profile("training_step"):
                training_step_output = self.trainer.accelerator.training_step(step_kwargs)
                self.trainer.accelerator.post_training_step()

            del step_kwargs

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            _check_training_step_output(self.trainer.lightning_module, training_step_output)

            result_collection, self._hiddens = _process_training_step_output(self.trainer, training_step_output)
            if result_collection is None:
                return None

        # accumulate loss. if accumulate_grad_batches==1, no effect
        closure_loss = result_collection.minimize / self.trainer.accumulate_grad_batches
        # the loss will get scaled for amp. avoid any modifications to it
        loss = closure_loss.detach().clone()
        return AttributeDict(closure_loss=closure_loss, loss=loss, result_collection=result_collection)

    def _track_and_norm_grad(self, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Tracks gradient norms and clips the gradients of all parameters optimized by the current optimizer.

        Args:
            optimizer: the current optimizer
        """
        # track gradient norms
        grad_norm_dict = {}
        can_log = (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0
        should_track = float(self.trainer.track_grad_norm) > 0
        if should_track and can_log:
            grad_norm_dict = grad_norm(self.trainer.lightning_module, self.trainer.track_grad_norm)

        # clip gradients
        self.trainer.accelerator.clip_gradients(
            optimizer, self.trainer.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm
        )
        return grad_norm_dict
