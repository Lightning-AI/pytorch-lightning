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

from contextlib import contextmanager, suppress
from copy import copy, deepcopy
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.core.step_result import Result
from pytorch_lightning.plugins import ParallelPlugin
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.trainer.supporters import TensorRunningAccum
from pytorch_lightning.utilities import _TPU_AVAILABLE, AMPType, DeviceType, parsing
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.finite_checks import detect_nan_parameters
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.warnings import WarningCache


class TrainLoop:

    def __init__(self, trainer, multiple_trainloader_mode: str):
        self.trainer = trainer
        self.accumulated_loss = None
        self.warning_cache = WarningCache()
        self._teardown_already_run = False
        self.running_loss = TensorRunningAccum(window_length=20)
        self.automatic_optimization = True
        self._curr_step_result = None
        self._cur_grad_norm_dict = None
        self._multiple_trainloader_mode = multiple_trainloader_mode
        self._skip_backward = False
        self.trainer._multiple_trainloader_mode = multiple_trainloader_mode

    def on_trainer_init(
        self,
        max_epochs: Optional[int],
        min_epochs: Optional[int],
        max_steps: Optional[int],
        min_steps: Optional[int],
        num_sanity_val_steps: int,
    ) -> None:
        self.trainer.global_step = 0
        self.trainer.current_epoch = 0
        self.trainer.should_stop = False
        self.trainer._state = TrainerState.INITIALIZING

        self.trainer.total_batch_idx = 0
        self.trainer.batch_idx = 0
        self.trainer.num_training_batches = 0
        self.trainer.train_dataloader = None

        # If neither max_epochs or max_steps is set, then use existing default of max_epochs = 1000
        self.trainer.max_epochs = 1000 if (max_epochs is None and max_steps is None) else max_epochs
        # If neither min_epochs or min_steps is set, then use existing default of min_epochs = 1
        self.trainer.min_epochs = 1 if (min_epochs is None and min_steps is None) else min_epochs
        self.trainer.max_steps = max_steps
        self.trainer.min_steps = min_steps

        if num_sanity_val_steps == -1:
            self.trainer.num_sanity_val_steps = float("inf")
        else:
            self.trainer.num_sanity_val_steps = num_sanity_val_steps

    @property
    def num_optimizers(self):
        num_optimizers = len(self.get_optimizers_iterable())
        return num_optimizers

    def should_skip_training(self):
        should_by_max_steps = self.trainer.max_steps is not None and self.trainer.global_step >= self.trainer.max_steps
        should_by_epoch = self.trainer.max_epochs is not None and self.trainer.current_epoch >= self.trainer.max_epochs
        return should_by_max_steps or should_by_epoch or self.trainer.num_training_batches == 0

    def on_train_start(self):
        # hook
        self.trainer.call_hook("on_train_start")

    def setup_fit(self, model, train_dataloader=None, val_dataloaders=None, datamodule=None):
        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        # links data to the trainer
        self.trainer.data_connector.attach_data(model, train_dataloader, val_dataloaders, datamodule)

        # check that model is configured correctly
        self.trainer.config_validator.verify_loop_configurations(model)

        # attach model log function to callback
        self.trainer.callback_connector.attach_model_logging_functions(model)

    def on_train_end(self):
        if self._teardown_already_run:
            return
        self._teardown_already_run = True

        # trigger checkpoint check. need to temporarily decrease the global step to avoid saving duplicates
        # when a checkpoint was saved at the last step
        self.trainer.global_step -= 1
        self.check_checkpoint_callback(should_update=True, is_last=True)
        self.trainer.global_step += 1

        # hook
        self.trainer.call_hook("on_train_end")

        # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
        # It might be related to xla tensors blocked when moving the cpu
        # kill loggers
        if self.trainer.logger is not None:
            self.trainer.logger.finalize("success")

        # summarize profile results
        self.trainer.profiler.describe()

        # give accelerators a chance to finish
        self.trainer.accelerator.on_train_end()

        # reset bookkeeping
        self.trainer._running_stage = None

    def check_checkpoint_callback(self, should_update, is_last=False):
        # TODO bake this logic into the ModelCheckpoint callback
        if should_update and self.trainer.checkpoint_connector.has_trained:
            callbacks = self.trainer.checkpoint_callbacks

            if is_last and any(cb.save_last and cb.verbose for cb in callbacks):
                rank_zero_info("Saving latest checkpoint...")

            model = self.trainer.lightning_module

            for cb in callbacks:
                cb.on_validation_end(self.trainer, model)

    def check_early_stopping_callback(self, should_update):
        # TODO bake this logic into the EarlyStopping callback
        if should_update and self.trainer.checkpoint_connector.has_trained:
            callbacks = [c for c in self.trainer.callbacks if isinstance(c, EarlyStopping)]
            model = self.trainer.lightning_module

            for cb in callbacks:
                cb.on_validation_end(self.trainer, model)

    def on_train_epoch_start(self, epoch):

        # update training progress in trainer
        self.trainer.current_epoch = epoch

        model = self.trainer.lightning_module

        # reset train dataloader
        if epoch != 0 and self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # todo: specify the possible exception
        with suppress(Exception):
            # set seed for distributed sampler (enables shuffling for each epoch)
            self.trainer.train_dataloader.sampler.set_epoch(epoch)

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_train_epoch_start(self.trainer, self.trainer.lightning_module)

        # stores accumulated grad fractions per batch
        self.accumulated_loss = TensorRunningAccum(window_length=self.trainer.accumulate_grad_batches)

        # hook
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")

    def on_train_batch_end(self, epoch_output, batch_end_outputs, batch, batch_idx, dataloader_idx):
        batch_end_outputs = [opt_idx_out for opt_idx_out in batch_end_outputs if len(opt_idx_out)]

        processed_batch_end_outputs = TrainLoop._prepare_outputs(batch_end_outputs, batch_mode=True)

        # hook
        self.trainer.call_hook('on_train_batch_end', processed_batch_end_outputs, batch, batch_idx, dataloader_idx)
        self.trainer.call_hook('on_batch_end')

        # figure out what to track for epoch end
        self.track_epoch_end_reduce_metrics(epoch_output, batch_end_outputs)

        # reset batch logger internals
        self.trainer.logger_connector.on_train_batch_end()

    def reset_train_val_dataloaders(self, model):
        if self.trainer.train_dataloader is None or not self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        if self.trainer.val_dataloaders is None and not self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_val_dataloader(model)

    def track_epoch_end_reduce_metrics(self, epoch_output, batch_end_outputs):

        # track the outputs to reduce at the end of the epoch
        for opt_idx, opt_outputs in enumerate(batch_end_outputs):
            sample_output = opt_outputs[-1]

            # decide if we need to reduce at the end of the epoch automatically
            auto_reduce_tng_result = isinstance(sample_output, Result) and sample_output.should_reduce_on_epoch_end
            hook_overridden = (
                is_overridden("training_epoch_end", model=self.trainer.lightning_module)
                or is_overridden("on_train_epoch_end", model=self.trainer.lightning_module)
            )

            # only track when a) it needs to be autoreduced OR b) the user wants to manually reduce on epoch end
            if not (hook_overridden or auto_reduce_tng_result):
                continue

            # with 1 step (no tbptt) don't use a sequence at epoch end
            if isinstance(opt_outputs, list) and len(opt_outputs) == 1 and not isinstance(opt_outputs[0], Result):
                opt_outputs = opt_outputs[0]

            epoch_output[opt_idx].append(opt_outputs)

    def get_optimizers_iterable(self):
        """
        Generates an iterable with (idx, optimizer) for each optimizer.
        """
        if not self.trainer.optimizer_frequencies:
            # call training_step once per optimizer
            return list(enumerate(self.trainer.optimizers))

        optimizer_freq_cumsum = np.cumsum(self.trainer.optimizer_frequencies)
        optimizers_loop_length = optimizer_freq_cumsum[-1]
        current_place_in_loop = self.trainer.total_batch_idx % optimizers_loop_length

        # find optimzier index by looking for the first {item > current_place} in the cumsum list
        opt_idx = np.argmax(optimizer_freq_cumsum > current_place_in_loop)
        return [[opt_idx, self.trainer.optimizers[opt_idx]]]

    def on_after_backward(self, training_step_output, batch_idx, untouched_loss):
        training_step_output.detach()

        # insert after step hook
        self.trainer.call_hook("on_after_backward")

        # when in dev debugging track the losses
        self.trainer.dev_debugger.track_train_loss_history(batch_idx, untouched_loss.detach())

    def _check_training_step_output(self, training_step_output):
        if isinstance(training_step_output, torch.Tensor) and not self.automatic_optimization:
            if training_step_output.grad_fn is None:
                # TODO: Find why - RuntimeError: Expected to mark a variable ready only once ...
                raise MisconfigurationException("In manual optimization, `training_step` should not return a Tensor")

    def training_step(self, split_batch, batch_idx, opt_idx, hiddens):
        # give the PL module a result for logging
        model_ref = self.trainer.lightning_module

        with self.trainer.profiler.profile("model_forward"):
            args = self.build_train_args(split_batch, batch_idx, opt_idx, hiddens)

            # manually capture logged metrics
            model_ref._current_fx_name = 'training_step'
            model_ref._results = Result()
            with self.trainer.profiler.profile("training_step"):
                training_step_output = self.trainer.accelerator.training_step(args)
                self.trainer.accelerator.post_training_step()

            self.trainer.logger_connector.cache_logged_metrics()

            self._check_training_step_output(training_step_output)

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            training_step_output_for_epoch_end, training_step_output = self._process_training_step_output(
                training_step_output, split_batch
            )
            if training_step_output_for_epoch_end is None:
                return

        # enable empty loss when using manual opt
        closure_loss = None
        untouched_loss = None

        if self.automatic_optimization:
            # accumulate loss. if accumulate_grad_batches==1, no effect
            closure_loss = training_step_output.minimize / self.trainer.accumulate_grad_batches

            # the loss will get scaled for amp. avoid any modifications to it
            untouched_loss = closure_loss.detach().clone()

        # result
        result = AttributeDict(
            closure_loss=closure_loss,
            loss=untouched_loss,
            training_step_output=training_step_output,
            training_step_output_for_epoch_end=training_step_output_for_epoch_end,
        )
        return result

    def _process_training_step_output(self, training_step_output, split_batch):
        training_step_output_for_epoch_end = training_step_output

        # enable validation_step return None
        if training_step_output_for_epoch_end is None:
            return None, None

        result = self.trainer.lightning_module._results

        loss = None
        hiddens = None
        result["extra"] = {}

        # handle dict return
        if isinstance(training_step_output, dict):
            loss = training_step_output.pop("loss", None)
            hiddens = training_step_output.pop("hiddens", None)
            if hiddens is not None:
                hiddens = hiddens.detach()
            result["extra"] = training_step_output

        # handle scalar return
        elif isinstance(training_step_output, torch.Tensor):
            loss = training_step_output

        # map to results under the hood
        result.minimize = loss
        self.trainer.hiddens = hiddens

        # track batch for manual reduction with result
        result.track_batch_size(len(split_batch))

        # track metrics without grads for epoch reduction
        training_step_output_for_epoch_end = copy(result)
        training_step_output_for_epoch_end = training_step_output_for_epoch_end.detach()
        if self.trainer.move_metrics_to_cpu:
            training_step_output_for_epoch_end = training_step_output_for_epoch_end.cpu()

        return training_step_output_for_epoch_end, result

    @staticmethod
    def _prepare_outputs(
        outputs: List[List[List[Result]]],
        batch_mode: bool,
    ) -> Union[List[List[List[Dict]]], List[List[Dict]], List[Dict], Dict]:
        """
        Extract required information from batch or epoch end results.

        Args:
            outputs: A 3-dimensional list of ``Result`` objects with dimensions:
                [optimizer outs][batch outs][tbptt steps].

            batch_mode: If True, ignore the batch output dimension.

        Returns:
            The cleaned outputs with ``Result`` objects converted to dictionaries. All list dimensions of size one will
            be collapsed.
        """
        processed_outputs = []
        for opt_outputs in outputs:
            # handle an edge case where an optimizer output is the empty list
            if len(opt_outputs) == 0:
                continue

            processed_batch_outputs = []

            if batch_mode:
                opt_outputs = [opt_outputs]

            for batch_outputs in opt_outputs:
                processed_tbptt_outputs = []

                for tbptt_output in batch_outputs:
                    out = tbptt_output.extra
                    out['loss'] = tbptt_output.minimize
                    processed_tbptt_outputs.append(out)

                # if there was only one tbptt step then we can collapse that dimension
                if len(processed_tbptt_outputs) == 1:
                    processed_tbptt_outputs = processed_tbptt_outputs[0]
                processed_batch_outputs.append(processed_tbptt_outputs)

            # batch_outputs should be just one dict (or a list of dicts if using tbptt) per optimizer
            if batch_mode:
                processed_batch_outputs = processed_batch_outputs[0]
            processed_outputs.append(processed_batch_outputs)

        # if there is only one optimiser then we collapse that dimension
        if len(processed_outputs) == 1:
            processed_outputs = processed_outputs[0]
        return processed_outputs

    def optimizer_step(self, optimizer, opt_idx, batch_idx, train_step_and_backward_closure):
        model_ref = self.trainer.lightning_module

        is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
        using_native_amp = self.trainer.amp_backend == AMPType.NATIVE

        # native amp + lbfgs is a no go right now
        if using_native_amp and is_lbfgs:
            raise MisconfigurationException(
                'native PyTorch amp and lbfgs are not compatible.'
                ' To request, please file a Github issue in PyTorch and tag @mcarilli'
            )

        # wraps into LightningOptimizer only for running step
        optimizer = LightningOptimizer._to_lightning_optimizer(optimizer, self.trainer, opt_idx)

        # model hook
        model_ref.optimizer_step(
            self.trainer.current_epoch,
            batch_idx,
            optimizer,
            opt_idx,
            train_step_and_backward_closure,
            on_tpu=self.trainer._device_type == DeviceType.TPU and _TPU_AVAILABLE,
            using_native_amp=using_native_amp,
            using_lbfgs=is_lbfgs,
        )

    def on_before_zero_grad(self, optimizer):
        self.trainer.call_hook('on_before_zero_grad', optimizer)

    def optimizer_zero_grad(self, batch_idx, optimizer, opt_idx):
        self.trainer.accelerator.optimizer_zero_grad(self.trainer.current_epoch, batch_idx, optimizer, opt_idx)

    def track_and_norm_grad(self, optimizer):
        # track gradient norms
        grad_norm_dic = self._track_gradient_norm()

        # clip gradients
        self.trainer.accelerator.clip_gradients(
            optimizer, self.trainer.gradient_clip_val, gradient_clip_algorithm=self.trainer.gradient_clip_algorithm
        )
        self._cur_grad_norm_dict = grad_norm_dic

    def _track_gradient_norm(self):
        grad_norm_dict = {}
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if float(self.trainer.track_grad_norm) > 0:
                model = self.trainer.lightning_module
                grad_norm_dict = model.grad_norm(self.trainer.track_grad_norm)
        return grad_norm_dict

    def tbptt_split_batch(self, batch):
        splits = [batch]
        if self.trainer.truncated_bptt_steps is not None:
            model_ref = self.trainer.lightning_module
            with self.trainer.profiler.profile("tbptt_split_batch"):
                splits = model_ref.tbptt_split_batch(batch, self.trainer.truncated_bptt_steps)
        return splits

    def run_training_epoch(self):
        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.trainer.accelerator.process_dataloader(self.trainer.train_dataloader)

        # track epoch output
        epoch_output = [[] for _ in range(self.num_optimizers)]

        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)
        dataloader_idx = 0
        val_loop_called = False

        for batch_idx, (batch, is_last_batch) in train_dataloader:

            self.trainer.batch_idx = batch_idx
            self.trainer.is_last_batch = is_last_batch

            # ------------------------------------
            # TRAINING_STEP + TRAINING_STEP_END
            # ------------------------------------
            with self.trainer.profiler.profile("run_training_batch"):
                batch_output = self.run_training_batch(batch, batch_idx, dataloader_idx)

            # when returning -1 from train_step, we end epoch early
            if batch_output.signal == -1:
                break

            # hook
            # TODO: add outputs to batches
            self.on_train_batch_end(
                epoch_output,
                batch_output.training_step_output_for_epoch_end,
                batch,
                batch_idx,
                dataloader_idx,
            )

            # -----------------------------------------
            # SAVE METRICS TO LOGGERS
            # -----------------------------------------
            self.trainer.logger_connector.log_train_step_metrics(batch_output)

            # -----------------------------------------
            # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
            # -----------------------------------------
            should_check_val = self.should_check_val_fx(batch_idx, is_last_batch)
            if should_check_val:
                self.trainer.validating = True
                self.trainer.run_evaluation()
                self.trainer.training = True
                val_loop_called = True

            # -----------------------------------------
            # SAVE LOGGERS (ie: Tensorboard, etc...)
            # -----------------------------------------
            self.save_loggers_on_train_batch_end()

            # update LR schedulers
            monitor_metrics = deepcopy(self.trainer.logger_connector.callback_metrics)
            self.update_train_loop_lr_schedulers(monitor_metrics=monitor_metrics)
            self.trainer.checkpoint_connector.has_trained = True

            # max steps reached, end training
            if (
                self.trainer.max_steps is not None and self.trainer.max_steps == self.trainer.global_step + 1
                and self._accumulated_batches_reached()
            ):
                break

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if self.trainer.should_stop:
                break

            self.trainer.total_batch_idx += 1

            # stop epoch if we limited the number of training batches
            if self._num_training_batches_reached(is_last_batch):
                break

            # progress global step according to grads progress
            self.increment_accumulated_grad_global_step()

        # handle epoch_output on epoch end
        self.on_train_epoch_end(epoch_output)

        # log epoch metrics
        self.trainer.logger_connector.log_train_epoch_end_metrics(epoch_output)

        should_check_val = self.should_check_val_fx(batch_idx, is_last_batch, on_epoch=True)
        should_skip_eval = self.trainer.evaluation_loop.should_skip_evaluation(self.trainer.num_val_batches)
        should_train_only = self.trainer.disable_validation or should_skip_eval

        # update epoch level lr_schedulers if no val loop outside train loop is triggered
        if (val_loop_called and not should_check_val) or should_train_only:
            self.trainer.optimizer_connector.update_learning_rates(interval='epoch')

        if should_train_only:
            self.check_checkpoint_callback(True)
            self.check_early_stopping_callback(True)

        if should_check_val:
            self.trainer.validating = True
            self.trainer.run_evaluation(on_epoch=True)
            self.trainer.training = True

        # increment the global step once
        # progress global step according to grads progress
        self.increment_accumulated_grad_global_step()

    def on_train_epoch_end(self, epoch_output: List[List[List[Result]]]) -> None:
        # inform logger the batch loop has finished
        self.trainer.logger_connector.on_train_epoch_end()

        # prepare epoch output
        processed_epoch_output = TrainLoop._prepare_outputs(epoch_output, batch_mode=False)

        # get the model and call model.training_epoch_end
        model = self.trainer.lightning_module

        if is_overridden('training_epoch_end', model=model):
            # run training_epoch_end
            # refresh the result for custom logging at the epoch level
            model._current_fx_name = 'training_epoch_end'

            # lightningmodule hook
            training_epoch_end_output = model.training_epoch_end(processed_epoch_output)

            if training_epoch_end_output is not None:
                raise MisconfigurationException(
                    'training_epoch_end expects a return of None. '
                    'HINT: remove the return statement in training_epoch_end'
                )

            # capture logging
            self.trainer.logger_connector.cache_logged_metrics()

        # call train epoch end hooks
        self.trainer.call_hook('on_train_epoch_end', processed_epoch_output)
        self.trainer.call_hook('on_epoch_end')

    def run_training_batch(self, batch, batch_idx, dataloader_idx):
        # track grad norms
        grad_norm_dic = {}

        # bookkeeping
        self.trainer.hiddens = None

        optimizers = self.prepare_optimizers()

        # track all outputs across time and num of optimizers
        batch_outputs = [[] for _ in range(len(optimizers))]

        if batch is None:
            return AttributeDict(signal=0, grad_norm_dic=grad_norm_dic)

        # hook
        response = self.trainer.call_hook("on_batch_start")
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic=grad_norm_dic)

        # hook
        response = self.trainer.call_hook("on_train_batch_start", batch, batch_idx, dataloader_idx)
        if response == -1:
            return AttributeDict(signal=-1, grad_norm_dic=grad_norm_dic)

        # lightning module hook
        splits = self.tbptt_split_batch(batch)

        for split_idx, split_batch in enumerate(splits):

            # create an iterable for optimizers and loop over them
            for opt_idx, optimizer in optimizers:

                # toggle model params + set info to logger_connector
                self.run_train_split_start(split_idx, split_batch, opt_idx, optimizer)

                if self.should_accumulate():
                    # For gradient accumulation

                    # -------------------
                    # calculate loss (train step + train step end)
                    # -------------------

                    # automatic_optimization=True: perform dpp sync only when performing optimizer_step
                    # automatic_optimization=False: don't block synchronization here
                    with self.block_ddp_sync_behaviour():
                        self.training_step_and_backward(
                            split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens
                        )

                    batch_outputs = self._process_closure_result(
                        batch_outputs=batch_outputs,
                        opt_idx=opt_idx,
                    )

                # ------------------------------
                # BACKWARD PASS
                # ------------------------------
                # gradient update with accumulated gradients

                else:
                    if self.automatic_optimization:

                        def train_step_and_backward_closure():
                            result = self.training_step_and_backward(
                                split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens
                            )
                            return None if result is None else result.loss

                        # optimizer step
                        self.optimizer_step(optimizer, opt_idx, batch_idx, train_step_and_backward_closure)

                    else:
                        self._curr_step_result = self.training_step(
                            split_batch, batch_idx, opt_idx, self.trainer.hiddens
                        )

                    if self._curr_step_result is None:
                        # user decided to skip optimization
                        # make sure to zero grad.
                        continue

                    batch_outputs = self._process_closure_result(
                        batch_outputs=batch_outputs,
                        opt_idx=opt_idx,
                    )

                    # todo: Properly aggregate grad_norm accros opt_idx and split_idx
                    grad_norm_dic = self._cur_grad_norm_dict
                    self._cur_grad_norm_dict = None

                    # update running loss + reset accumulated loss
                    self.update_running_loss()

        result = AttributeDict(
            signal=0,
            grad_norm_dic=grad_norm_dic,
            training_step_output_for_epoch_end=batch_outputs,
        )
        return result

    @contextmanager
    def block_ddp_sync_behaviour(self, should_block_sync: bool = False):
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
        if (
            isinstance(self.trainer.training_type_plugin, ParallelPlugin)
            and (self.automatic_optimization or should_block_sync)
        ):
            with self.trainer.training_type_plugin.block_backward_sync():
                yield None
        else:
            yield None

    def _process_closure_result(self, batch_outputs: list, opt_idx: int) -> list:
        opt_closure_result = self._curr_step_result

        if opt_closure_result is not None:

            # cache metrics
            self.trainer.logger_connector.cache_training_step_metrics(opt_closure_result)

            # check if loss or model weights are nan
            if self.trainer.terminate_on_nan:
                self._check_finite(opt_closure_result.loss)

            # track all the outputs across all steps
            batch_opt_idx = opt_idx if len(batch_outputs) > 1 else 0
            batch_outputs[batch_opt_idx].append(opt_closure_result.training_step_output_for_epoch_end)

            if self.automatic_optimization:
                # track total loss for logging (avoid mem leaks)
                self.accumulated_loss.append(opt_closure_result.loss)

        self._curr_step_result = None

        return batch_outputs

    def training_step_and_backward(self, split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """Wrap forward, zero_grad and backward in a closure so second order methods work"""
        with self.trainer.profiler.profile("training_step_and_backward"):
            # lightning module hook
            result = self.training_step(split_batch, batch_idx, opt_idx, hiddens)
            self._curr_step_result = result

            if not self._skip_backward and self.automatic_optimization:
                is_first_batch_to_accumulate = batch_idx % self.trainer.accumulate_grad_batches == 0

                if is_first_batch_to_accumulate:
                    self.on_before_zero_grad(optimizer)
                    self.optimizer_zero_grad(batch_idx, optimizer, opt_idx)

                # backward pass
                if result is not None:
                    with self.trainer.profiler.profile("backward"):
                        self.backward(result, optimizer, opt_idx)

                    # hook - call this hook only
                    # when gradients have finished to accumulate
                    if not self.should_accumulate():
                        self.on_after_backward(result.training_step_output, batch_idx, result.loss)

                    # check if loss or model weights are nan
                    if self.trainer.terminate_on_nan:
                        self._check_finite(result.loss)

                else:
                    self.warning_cache.warn("training_step returned None if it was on purpose, ignore this warning...")

                if len(self.trainer.optimizers) > 1:
                    # revert back to previous state
                    self.trainer.lightning_module.untoggle_optimizer(opt_idx)

        return result

    def _check_finite(self, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss).all():
            raise ValueError(f'The loss returned in `training_step` is {loss}.')
        model = self.trainer.lightning_module
        detect_nan_parameters(model)

    def backward(self, result, optimizer, opt_idx, *args, **kwargs):
        self.trainer.dev_debugger.track_event("backward_call")

        should_accumulate = self.should_accumulate()

        # backward can be called manually in the training loop
        if isinstance(result, torch.Tensor):
            self.trainer.accelerator.backward(result, optimizer, opt_idx, should_accumulate, *args, **kwargs)
        else:
            result.closure_loss = self.trainer.accelerator.backward(
                result.closure_loss, optimizer, opt_idx, should_accumulate, *args, **kwargs
            )

        if not self.should_accumulate():
            # track gradients
            self.track_and_norm_grad(optimizer=optimizer)

    def update_train_loop_lr_schedulers(self, monitor_metrics=None):
        num_accumulated_batches_reached = self._accumulated_batches_reached()
        num_training_batches_reached = self._num_training_batches_reached()

        if num_accumulated_batches_reached or num_training_batches_reached:
            # update lr
            self.trainer.optimizer_connector.update_learning_rates(interval="step", monitor_metrics=monitor_metrics)

    def increment_accumulated_grad_global_step(self):
        num_accumulated_batches_reached = self._accumulated_batches_reached()
        num_training_batches_reached = self._num_training_batches_reached()

        # progress global step according to grads progress
        if num_accumulated_batches_reached or num_training_batches_reached:
            self.trainer.global_step = self.trainer.accelerator.update_global_step(
                self.trainer.total_batch_idx, self.trainer.global_step
            )

    def _accumulated_batches_reached(self):
        return (self.trainer.batch_idx + 1) % self.trainer.accumulate_grad_batches == 0

    def _num_training_batches_reached(self, is_last_batch=False):
        return (self.trainer.batch_idx + 1) == self.trainer.num_training_batches or is_last_batch

    def should_accumulate(self):
        # checks if backward or backward + optimizer step (via closure)
        accumulation_done = self._accumulated_batches_reached()
        is_final_batch = self._num_training_batches_reached()
        return not (accumulation_done or is_final_batch)

    def should_check_val_fx(self, batch_idx, is_last_batch, on_epoch=False):
        # decide if we should run validation
        is_val_check_batch = (batch_idx + 1) % self.trainer.val_check_batch == 0
        is_val_check_epoch = (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        can_check_val = self.trainer.enable_validation and is_val_check_epoch
        is_last_batch_for_infinite_dataset = is_last_batch and self.trainer.val_check_batch == float("inf")
        epoch_end_val_check = (batch_idx + 1) % self.trainer.num_training_batches == 0

        should_check_val = ((is_val_check_batch and epoch_end_val_check) or self.trainer.should_stop
                            or is_last_batch_for_infinite_dataset
                            ) if on_epoch else (is_val_check_batch and not epoch_end_val_check)

        return should_check_val and can_check_val

    def build_train_args(self, batch, batch_idx, opt_idx, hiddens):
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx]

        if len(self.trainer.optimizers) > 1:
            if self.trainer.has_arg("training_step", "optimizer_idx"):
                if not self.automatic_optimization:
                    self.warning_cache.warn(
                        "`training_step` hook signature has changed in v1.3."
                        " `optimizer_idx` argument has been removed in case of manual optimization. Support for"
                        " the old signature will be removed in v1.5", DeprecationWarning
                    )
                args.append(opt_idx)
            elif not self.trainer.has_arg("training_step", "optimizer_idx") and self.automatic_optimization:
                raise ValueError(
                    f"Your LightningModule defines {len(self.trainer.optimizers)} optimizers but"
                    ' `training_step` is missing the `optimizer_idx` argument.'
                )

        # pass hiddens if using tbptt
        if self.trainer.truncated_bptt_steps is not None:
            args.append(hiddens)

        return args

    def save_loggers_on_train_batch_end(self):
        # when loggers should save to disk
        should_flush_logs = self.trainer.logger_connector.should_flush_logs
        if should_flush_logs and self.trainer.is_global_zero and self.trainer.logger is not None:
            self.trainer.logger.save()

    def prepare_optimizers(self):
        # in manual optimization we loop over all optimizers at once
        optimizers = self.get_optimizers_iterable()
        if not self.automatic_optimization:
            optimizers = [optimizers[0]]
        return optimizers

    def run_train_split_start(self, split_idx, split_batch, opt_idx, optimizer):
        # set split_idx to trainer for tracking
        self.trainer.split_idx = split_idx

        # make sure only the gradients of the current optimizer's parameters are calculated
        # in the training step to prevent dangling gradients in multiple-optimizer setup.
        if self.automatic_optimization and len(self.trainer.optimizers) > 1:
            model = self.trainer.lightning_module
            model.toggle_optimizer(optimizer, opt_idx)

        # use to track metrics internally
        self.trainer.logger_connector.on_train_split_start(split_idx, opt_idx, split_batch)

    def update_running_loss(self):
        accumulated_loss = self.accumulated_loss.mean()

        if accumulated_loss is not None:
            # calculate running loss for display
            self.running_loss.append(self.accumulated_loss.mean() * self.trainer.accumulate_grad_batches)

        # reset for next set of accumulated grads
        self.accumulated_loss.reset()
