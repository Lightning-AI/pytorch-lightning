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

from collections import OrderedDict
from contextlib import contextmanager
from copy import copy
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, Tuple

import numpy as np
import torch
from deprecate import void
from torch import Tensor
from torch.optim import Optimizer

from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.plugins import ParallelPlugin
from pytorch_lightning.trainer.connectors.logger_connector.result import ResultCollection
from pytorch_lightning.trainer.progress import OptimizationProgress
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import AMPType, AttributeDict, DeviceType, grad_norm
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
from pytorch_lightning.utilities.imports import _TPU_AVAILABLE
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.utilities.warnings import WarningCache


class TrainingBatchLoop(Loop):
    """Runs over a single batch of data."""

    def __init__(self) -> None:
        super().__init__()
        self.accumulated_loss: Optional[Tensor] = None
        self.batch_outputs: Optional[List[List[STEP_OUTPUT]]] = None
        self.running_loss: TensorRunningAccum = TensorRunningAccum(window_length=20)
        self.batch_idx: int = 0
        self.split_idx: Optional[int] = None
        self.optim_progress = OptimizationProgress()

        self._warning_cache: WarningCache = WarningCache()
        self._hiddens: Optional[Tensor] = None
        self._optimizer_freq_cumsum: Optional[int] = None
        self._remaining_splits: Optional[List[Any]] = None
        self._skip_backward: bool = False

    @property
    def done(self) -> bool:
        """Returns if all batch splits have been processed already"""
        return len(self._remaining_splits) == 0

    @property
    def optimizer_freq_cumsum(self) -> int:
        """Returns the cumulated sum of optimizer frequencies"""
        if self._optimizer_freq_cumsum is None:
            self._optimizer_freq_cumsum = np.cumsum(self.trainer.optimizer_frequencies)
        return self._optimizer_freq_cumsum

    def connect(self, **kwargs: "Loop") -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not connect any child loops.")

    def run(self, batch: Any, batch_idx: int, dataloader_idx: int) -> AttributeDict:
        """Runs all the data splits and the ``on_batch_start`` and ``on_train_batch_start`` hooks

        Args:
            batch: the current batch to run the train step on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
        """
        if batch is None:
            self._warning_cache.warn("train_dataloader yielded None. If this was on purpose, ignore this warning...")
            return AttributeDict(signal=0, training_step_output=[[]])

        # hook
        self.trainer.logger_connector.on_batch_start()
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1)

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, dataloader_idx)
        if response == -1:
            return AttributeDict(signal=-1)

        self.trainer.fit_loop.epoch_loop.batch_progress.increment_started()

        super().run(batch, batch_idx, dataloader_idx)
        output = AttributeDict(signal=0, training_step_output=self.batch_outputs)
        self.batch_outputs = None  # free memory
        return output

    def reset(self) -> None:
        """Resets the loop state"""
        self._hiddens = None
        self.batch_idx = 0
        self.batch_outputs = [[] for _ in range(len(self.trainer.optimizers))]

    def on_run_start(self, batch: Any, batch_idx: int, dataloader_idx: int):
        """Splits the data into tbptt splits

        Args:
            batch: the current batch to run the trainstep on
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
        """
        void(batch_idx, dataloader_idx)
        self._remaining_splits = list(enumerate(self._tbptt_split_batch(batch)))

    def advance(self, batch, batch_idx, dataloader_idx):
        """Runs the train step together with optimization (if necessary) on the current batch split

        Args:
            batch: the current batch to run the training on (this is not the split!)
            batch_idx: the index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch
        """
        void(batch, dataloader_idx)
        split_idx, split_batch = self._remaining_splits.pop(0)
        self.batch_idx = batch_idx
        self.split_idx = split_idx

        # let logger connector extract current batch size
        self.trainer.logger_connector.on_train_split_start(batch_idx, split_idx, split_batch)

        if self.trainer.lightning_module.automatic_optimization:
            for opt_idx, optimizer in self.get_active_optimizers(batch_idx):
                # handle optimization restart
                if self.restarting:
                    if opt_idx < self.optim_progress.optimizer_idx:
                        continue

                self.optim_progress.optimizer_idx = opt_idx

                result = self._run_optimization(batch_idx, split_batch, opt_idx, optimizer)
                if result:
                    self.batch_outputs[opt_idx].append(copy(result.training_step_output))
        else:
            # in manual optimization, there is no looping over optimizers
            result = self._run_optimization(batch_idx, split_batch)
            if result:
                self.batch_outputs[0].append(copy(result.training_step_output))

    def teardown(self) -> None:
        # release memory
        self._remaining_splits = None

    def num_active_optimizers(self, batch_idx: Optional[int] = None) -> int:
        """Gets the number of active optimizers based on their frequency"""
        return len(self.get_active_optimizers(batch_idx))

    def _run_optimization(
        self, batch_idx: int, split_batch: Any, opt_idx: int = 0, optimizer: Optional[torch.optim.Optimizer] = None
    ):
        """Runs closure (train step + backward) together with optimization if necessary.

        Args:
            batch_idx: the index of the current batch
            split_batch: the current tbptt split of the whole batch
            opt_idx: the index of the current optimizer
            optimizer: the current optimizer
        """
        # TODO(@awaelchli): In v1.5, when optimizer_idx gets removed from training_step in manual_optimization, change
        #   opt_idx=0 to opt_idx=None in the signature here

        # toggle model params
        self._run_optimization_start(opt_idx, optimizer)

        result = AttributeDict()
        closure = self._make_closure(split_batch, batch_idx, opt_idx, optimizer, self._hiddens, result)

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
            if self.trainer.lightning_module.automatic_optimization:
                self._optimizer_step(optimizer, opt_idx, batch_idx, closure)
            else:
                result = self._training_step(split_batch, batch_idx, opt_idx, self._hiddens)

        if result:
            # if no result, user decided to skip optimization
            # otherwise update running loss + reset accumulated loss
            self._update_running_loss(result.loss)
            self._process_closure_result(result)

        # untoggle model params
        self._run_optimization_end(opt_idx)
        return result

    def _training_step_and_backward_closure(
        self,
        split_batch: Any,
        batch_idx: int,
        opt_idx: int,
        optimizer: Optimizer,
        hiddens: Tensor,
        return_result: AttributeDict,
    ) -> Optional[Tensor]:
        """Closure for training step and backward

        Args:
            split_batch: the current tbptt split of the batch
            batch_idx: the index of the current batch
            opt_idx: the index of the current optimizer
            optimizer: the current optimizer
            hiddens: the hidden state of the recurrent net
            return_result: the storage of the trainstep results
        """

        result = self.training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, hiddens)
        if result is not None:
            return_result.update(result)
            return return_result.loss

    def _make_closure(self, *closure_args: Any, **closure_kwargs: Any) -> Callable:
        """Wraps the training step closure into a partial object which will be called within ``optimizer.step``."""
        partial_func = partial(self._training_step_and_backward_closure, *closure_args, **closure_kwargs)
        return update_wrapper(partial_func, self._training_step_and_backward_closure)

    def _process_closure_result(self, opt_closure_result: Optional[AttributeDict]) -> None:
        """Checks if the closure results is finite and optionally breaks if it is not

        Args:
            opt_closure_result: the result of the train step wrapped in an attribute dict
        """
        if not opt_closure_result:
            return

        # check if loss or model weights are nan
        if self.trainer.terminate_on_nan:
            self._check_finite(opt_closure_result.loss)

    def _check_training_step_output(self, training_step_output: STEP_OUTPUT) -> None:
        """Sanity checks that training produced a valid output and optimizer step has already been called in manual
        optimization.

        Args:
            training_step_output: the output of the training step (before wrapping in an AttributeDict)

        """
        if isinstance(training_step_output, Tensor) and not self.trainer.lightning_module.automatic_optimization:
            if training_step_output.grad_fn is None:
                # TODO: Find why - RuntimeError: Expected to mark a variable ready only once ...
                raise MisconfigurationException("In manual optimization, `training_step` should not return a Tensor")
        elif self.trainer.lightning_module.automatic_optimization:
            if not any(
                (
                    isinstance(training_step_output, Tensor),
                    (isinstance(training_step_output, Mapping) and "loss" in training_step_output),
                    training_step_output is None,
                )
            ):
                raise MisconfigurationException(
                    "In automatic optimization, `training_step` must either return a Tensor, "
                    "a dict with key 'loss' or None (where the step will be skipped)."
                )

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
            step_kwargs = self._build_kwargs(split_batch, batch_idx, opt_idx, hiddens)

            # manually capture logged metrics
            model_ref._current_fx_name = "training_step"
            with self.trainer.profiler.profile("training_step"):
                training_step_output = self.trainer.accelerator.training_step(step_kwargs)
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

    def _process_training_step_output(self, training_step_output: STEP_OUTPUT) -> Optional[ResultCollection]:
        """Adds the :param:`training_step_output` to the trainer's results

        Args:
            training_step_output: the output of the training step (before wrapping into an AttributeDict)

        Returns:
            the updated results if the training_step's output was not None else None
        """
        if training_step_output is None:
            return None

        results = self.trainer._results

        loss = None
        hiddens = None

        # handle dict return
        if isinstance(training_step_output, dict):
            # this should not modify the `training_step_output`, as the user could be using it after `training_step_end`
            loss = training_step_output.get("loss")
            hiddens = training_step_output.get("hiddens")
            # detach hiddens to avoid `RuntimeError: Trying to backward through the graph a second time`
            hiddens = apply_to_collection(hiddens, Tensor, lambda t: t.detach())
            # use the setter instead of `dict.update` because it calls `detach` on the tensor items
            results.extra = {k: v for k, v in training_step_output.items() if k not in ("loss", "hiddens")}

        # handle scalar return
        elif isinstance(training_step_output, Tensor):
            loss = training_step_output

        # map to results under the hood
        results.minimize = loss
        self._hiddens = hiddens

        if self.trainer.move_metrics_to_cpu:
            results.cpu()
        return results

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
        using_native_amp = self.trainer.amp_backend == AMPType.NATIVE

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

    def _track_and_norm_grad(self, optimizer: torch.optim.Optimizer) -> Dict[str, Tensor]:
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

    def _accumulated_batches_reached(self) -> bool:
        """Determine if accumulation will be finished by the end of the current batch."""
        # FIXME(@awaelchli): use progress tracking of batches instead of manual batch_idx
        return (self.batch_idx + 1) % self.trainer.accumulate_grad_batches == 0

    def _num_training_batches_reached(self, is_last_batch: bool = False) -> bool:
        """Checks whether sufficient training batches have been processed.

        Args:
            is_last_batch: Whether the current batch is the last one
        """
        # FIXME(@awaelchli): use progress tracking of batches instead of manual batch_idx
        return (self.batch_idx + 1) == self.trainer.num_training_batches or is_last_batch

    def should_accumulate(self) -> bool:
        """Checks if the optimizer step should be performed or gradients should be accumulated for the current step."""
        # checks if backward or backward + optimizer step (via closure)
        accumulation_done = self._accumulated_batches_reached()
        is_final_batch = self._num_training_batches_reached()
        return not (accumulation_done or is_final_batch)

    def _tbptt_split_batch(self, batch: Any) -> List[Any]:
        """Splits a single batch into a list of sequence steps for tbptt.

        Args:
            batch: the current batch to split
        """
        tbptt_steps = self._truncated_bptt_steps()
        if tbptt_steps == 0:
            return [batch]

        model_ref = self.trainer.lightning_module
        with self.trainer.profiler.profile("tbptt_split_batch"):
            splits = model_ref.tbptt_split_batch(batch, tbptt_steps)
        return splits

    def _run_optimization_start(self, opt_idx: int, optimizer: torch.optim.Optimizer) -> None:
        """Toggles the optimizer to ensure the correct one is used and prevend dangling grads.

        Args:
            opt_idx: the index of the optimizer to use
            optimizer: the optimizer to use

        """
        # make sure only the gradients of the current optimizer's parameters are calculated
        # in the training step to prevent dangling gradients in multiple-optimizer setup.
        if self.trainer.lightning_module.automatic_optimization and len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.toggle_optimizer(optimizer, opt_idx)

    def _run_optimization_end(self, opt_idx: int) -> None:
        if self.trainer.lightning_module.automatic_optimization and len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.untoggle_optimizer(opt_idx)

    @contextmanager
    def block_ddp_sync_behaviour(self, should_block_sync: bool = False) -> Generator[None, None, None]:
        """
        automatic_optimization = True
        Blocks ddp sync gradients behaviour on backwards pass.
        This is useful for skipping sync when accumulating gradients, reducing communication overhead

        automatic_optimization = False
        do not block ddp gradient sync when using manual optimization
        as gradients are needed within the training step

        Returns:
            context manager with sync behaviour off
        """
        if isinstance(self.trainer.training_type_plugin, ParallelPlugin) and (
            self.trainer.lightning_module.automatic_optimization or should_block_sync
        ):
            with self.trainer.training_type_plugin.block_backward_sync():
                yield None
        else:
            yield None

    def training_step_and_backward(
        self,
        split_batch: Any,
        batch_idx: int,
        opt_idx: int,
        optimizer: torch.optim.Optimizer,
        hiddens: Optional[Tensor],
    ) -> STEP_OUTPUT:
        """Wrap forward, zero_grad and backward in a closure so second order methods work"""
        with self.trainer.profiler.profile("training_step_and_backward"):
            # lightning module hook
            result = self._training_step(split_batch, batch_idx, opt_idx, hiddens)

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

    def _check_finite(self, loss: Tensor) -> None:
        """Checks fotr finite parameters and loss values.

        Args:
            loss: the loss value to check to be finite
        """
        if not torch.isfinite(loss).all():
            raise ValueError(f"The loss returned in `training_step` is {loss}.")
        model = self.trainer.lightning_module
        detect_nan_parameters(model)

    def backward(
        self, result: STEP_OUTPUT, optimizer: Optional[torch.optim.Optimizer], *args: Any, **kwargs: Any
    ) -> None:
        """Performs the backward step.

        Args:
            result: The output of the trainstep (including the loss value)
            optimizer: Current optimizer being used. ``None`` if using manual optimization.
            opt_idx: Index of the current optimizer being used. ``None`` if using manual optimization.
        """
        # backward can be called manually in the training loop
        if isinstance(result, Tensor):
            self.trainer.accelerator.backward(result, optimizer, *args, **kwargs)
        else:
            result.closure_loss = self.trainer.accelerator.backward(result.closure_loss, optimizer, *args, **kwargs)

        if not self.should_accumulate():
            # track gradients
            grad_norm_dict = self._track_and_norm_grad(optimizer=optimizer)
            if grad_norm_dict:
                self.trainer.lightning_module._current_fx_name = "on_after_backward"
                self.trainer.lightning_module.log_grad_norm(grad_norm_dict)

    def _update_running_loss(self, current_loss: Tensor) -> None:
        """Updates the running loss value with the current value"""
        if self.trainer.lightning_module.automatic_optimization:
            # track total loss for logging (avoid mem leaks)
            self.accumulated_loss.append(current_loss)

        accumulated_loss = self.accumulated_loss.mean()

        if accumulated_loss is not None:
            # calculate running loss for display
            self.running_loss.append(self.accumulated_loss.mean() * self.trainer.accumulate_grad_batches)

        # reset for next set of accumulated grads
        self.accumulated_loss.reset()

    def get_active_optimizers(self, batch_idx: Optional[int] = None) -> List[Tuple[int, Optimizer]]:
        """
        Returns the currently active optimizers. When multiple optimizers are used with different frequencies,
        only one of the optimizers is active at a time.

        Returns:
            A list of tuples (opt_idx, optimizer) of currently active optimizers.
        """
        if not self.trainer.optimizer_frequencies:
            # call training_step once per optimizer
            return list(enumerate(self.trainer.optimizers))

        optimizers_loop_length = self.optimizer_freq_cumsum[-1]
        current_place_in_loop = batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = int(np.argmax(self.optimizer_freq_cumsum > current_place_in_loop))
        return [(opt_idx, self.trainer.optimizers[opt_idx])]

    def _build_kwargs(self, batch: Any, batch_idx: int, opt_idx: int, hiddens: Optional[Tensor]) -> Dict[str, Any]:
        """Builds the keyword arguments for training_step

        Args:
            batch: the batch to train on
            batch_idx: the index of the current batch
            opt_idx: the index of the current optimizer
            hiddens: the hidden state of the previous RNN iteration

        Returns:
            the keyword arguments for the training step
        """
        # enable not needing to add opt_idx to training_step
        step_kwargs = OrderedDict([("batch", batch), ("batch_idx", batch_idx)])

        lightning_module = self.trainer.lightning_module

        if len(self.trainer.optimizers) > 1:
            training_step_fx = getattr(lightning_module, "training_step")
            has_opt_idx_in_train_step = is_param_in_hook_signature(training_step_fx, "optimizer_idx")
            if has_opt_idx_in_train_step:
                if not lightning_module.automatic_optimization:
                    self._warning_cache.deprecation(
                        "`training_step` hook signature has changed in v1.3."
                        " `optimizer_idx` argument has been removed in case of manual optimization. Support for"
                        " the old signature will be removed in v1.5"
                    )
                step_kwargs["optimizer_idx"] = opt_idx
            elif not has_opt_idx_in_train_step and lightning_module.automatic_optimization:
                raise ValueError(
                    f"Your LightningModule defines {len(self.trainer.optimizers)} optimizers but"
                    " `training_step` is missing the `optimizer_idx` argument."
                )

        # pass hiddens if using tbptt
        if self._truncated_bptt_enabled():
            step_kwargs["hiddens"] = hiddens

        return step_kwargs

    def _truncated_bptt_enabled(self) -> bool:
        """Temporary tbptt utilities until this flag is fully migrated to the lightning module."""
        return self._truncated_bptt_steps() > 0

    def _truncated_bptt_steps(self) -> int:
        """Returns the number of tbptt steps"""
        lightning_module = self.trainer.lightning_module
        # Give precedence to the LightningModule as the Trainer flag will be removed in v1.5
        if lightning_module.truncated_bptt_steps > 0:
            return lightning_module.truncated_bptt_steps
        return self.trainer.truncated_bptt_steps or 0
