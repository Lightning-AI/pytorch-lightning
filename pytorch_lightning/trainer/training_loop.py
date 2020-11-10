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

from contextlib import contextmanager
from copy import copy, deepcopy

import numpy as np
import torch
import torch.distributed as torch_distrib

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.core.memory import ModelSummary
from pytorch_lightning.core.step_result import EvalResult, Result
from pytorch_lightning.trainer.states import TrainerState
from pytorch_lightning.trainer.supporters import TensorRunningAccum, Accumulator
from pytorch_lightning.utilities import parsing, AMPType
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import recursive_detach
from pytorch_lightning.utilities.model_utils import is_overridden
from pytorch_lightning.utilities.parsing import AttributeDict
from pytorch_lightning.utilities.warning_utils import WarningCache


class TrainLoop:
    def __init__(self, trainer):
        self.trainer = trainer
        self.early_stopping_accumulator = None
        self.checkpoint_accumulator = None
        self.accumulated_loss = None
        self.warning_cache = WarningCache()
        self._teardown_already_run = False
        self.running_loss = TensorRunningAccum(window_length=20)
        self.automatic_optimization = True
        self._curr_step_result = None
        self._cur_grad_norm_dict = None

    def on_trainer_init(
        self, max_epochs, min_epochs, max_steps, min_steps, num_sanity_val_steps, automatic_optimization
    ):
        self.trainer.global_step = 0
        self.trainer.current_epoch = 0
        self.trainer.interrupted = False
        self.trainer.should_stop = False
        self.trainer._state = TrainerState.INITIALIZING

        self.trainer.total_batch_idx = 0
        self.trainer.batch_idx = 0
        self.trainer.num_training_batches = 0
        self.trainer.train_dataloader = None
        self.automatic_optimization = automatic_optimization

        self.trainer.max_epochs = max_epochs
        self.trainer.min_epochs = min_epochs
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
        if self.trainer.current_epoch >= self.trainer.max_epochs:
            return True

        if self.trainer.limit_train_batches == 0:
            return True

        return False

    def on_train_start(self):
        # clear cache before training
        if self.trainer.on_gpu and self.trainer.root_gpu is not None:
            # use context because of:
            # https://discuss.pytorch.org/t/out-of-memory-when-i-use-torch-cuda-empty-cache/57898
            with torch.cuda.device(f"cuda:{self.trainer.root_gpu}"):
                torch.cuda.empty_cache()

        # hook
        self.trainer.call_hook("on_train_start")

    def setup_fit(self, model, train_dataloader, val_dataloaders, datamodule):
        # bind logger and other properties
        self.trainer.model_connector.copy_trainer_model_properties(model)

        # clean hparams
        if hasattr(model, "hparams"):
            parsing.clean_namespace(model.hparams)

        # links data to the trainer
        self.trainer.data_connector.attach_data(model, train_dataloader, val_dataloaders, datamodule)

        # check that model is configured correctly
        self.trainer.config_validator.verify_loop_configurations(model)

    def setup_training(self, model: LightningModule):
        """Sanity check a few things before starting actual training.

        Args:
            model: The model to run sanity test on.
        """
        # --------------------------
        # Setup??
        # --------------------------
        ref_model = model
        if self.trainer.data_parallel:
            ref_model = model.module

        # set the ranks and devices
        self.trainer.accelerator_backend.dist.rank = self.trainer.global_rank
        self.trainer.accelerator_backend.dist.device = ref_model.device

        # give model convenience properties
        ref_model.trainer = self.trainer

        # set local properties on the model
        self.trainer.model_connector.copy_trainer_model_properties(ref_model)

        # init amp. Must be done here instead of __init__ to allow ddp to work
        if self.trainer.amp_backend == AMPType.NATIVE and self.trainer.precision == 16 and not self.trainer.use_tpu:
            self.trainer.scaler = torch.cuda.amp.GradScaler()

        # log hyper-parameters
        if self.trainer.logger is not None:
            # save exp to get started (this is where the first experiment logs are written)
            self.trainer.logger.log_hyperparams(ref_model.hparams_initial)
            self.trainer.logger.log_graph(ref_model)
            self.trainer.logger.save()

        # wait for all to join if on distributed
        self.trainer.accelerator_backend.barrier("setup_training")

        # register auto-resubmit when on SLURM
        self.trainer.slurm_connector.register_slurm_signal_handlers()

        # --------------------------
        # Pre-train
        # --------------------------
        # on pretrain routine start
        self.trainer.on_pretrain_routine_start(ref_model)
        if self.trainer.is_function_implemented("on_pretrain_routine_start"):
            ref_model.on_pretrain_routine_start()

        # print model summary
        if self.trainer.is_global_zero and self.trainer.weights_summary is not None and not self.trainer.testing:
            if self.trainer.weights_summary in ModelSummary.MODES:
                ref_model.summarize(mode=self.trainer.weights_summary)
            else:
                raise MisconfigurationException("weights_summary can be None, " + ", ".join(ModelSummary.MODES))

        # track model now.
        # if cluster resets state, the model will update with the saved weights
        self.trainer.model = model

        # restore training and model before hpc is called
        self.trainer.checkpoint_connector.restore_weights(model)

        # on pretrain routine end
        self.trainer.on_pretrain_routine_end(ref_model)
        if self.trainer.is_function_implemented("on_pretrain_routine_end"):
            ref_model.on_pretrain_routine_end()

    def on_train_end(self):
        if self._teardown_already_run:
            return

        self._teardown_already_run = True

        # trigger checkpoint check. need to temporarily decrease the global step to avoid saving duplicates
        # when a checkpoint was saved at the last step
        self.trainer.global_step -= 1
        self.check_checkpoint_callback(should_save=True, is_last=True)
        self.trainer.global_step += 1

        # hook
        self.trainer.call_hook("on_train_end")

        # kill loggers
        if self.trainer.logger is not None:
            self.trainer.logger.finalize("success")

        # summarize profile results
        if self.trainer.global_rank == 0:
            self.trainer.profiler.describe()

        # give accelerators a chance to finish
        self.trainer.accelerator_backend.on_train_end()

        # clear mem
        if self.trainer.on_gpu:
            model = self.trainer.get_model()
            model.cpu()
            torch.cuda.empty_cache()

    def check_checkpoint_callback(self, should_save, is_last=False):
        # TODO bake this logic into the checkpoint callback
        if should_save and self.trainer.checkpoint_connector.has_trained:
            checkpoint_callbacks = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
            if is_last and any(c.save_last for c in checkpoint_callbacks):
                rank_zero_info("Saving latest checkpoint...")
            model = self.trainer.get_model()
            [c.on_validation_end(self.trainer, model) for c in checkpoint_callbacks]

    def on_train_epoch_start(self, epoch):

        # update training progress in trainer
        self.trainer.current_epoch = epoch

        model = self.trainer.get_model()

        # reset train dataloader
        if self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        # set seed for distributed sampler (enables shuffling for each epoch)
        try:
            self.trainer.train_dataloader.sampler.set_epoch(epoch)
        except Exception:
            pass

        # changing gradient according accumulation_scheduler
        self.trainer.accumulation_scheduler.on_epoch_start(self.trainer, self.trainer.get_model())

        # stores accumulated grad fractions per batch
        self.accumulated_loss = TensorRunningAccum(window_length=self.trainer.accumulate_grad_batches)

        # structured result accumulators for callbacks
        self.early_stopping_accumulator = Accumulator()
        self.checkpoint_accumulator = Accumulator()

        # hook
        self.trainer.call_hook("on_epoch_start")
        self.trainer.call_hook("on_train_epoch_start")

    def on_train_batch_end(self, epoch_output, epoch_end_outputs, batch, batch_idx, dataloader_idx):
        # hook
        self.trainer.call_hook('on_batch_end')
        self.trainer.call_hook('on_train_batch_end', epoch_end_outputs, batch, batch_idx, dataloader_idx)

        # figure out what to track for epoch end
        self.track_epoch_end_reduce_metrics(epoch_output, epoch_end_outputs)

    def reset_train_val_dataloaders(self, model):
        if not self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_train_dataloader(model)

        if self.trainer.val_dataloaders is None and not self.trainer.reload_dataloaders_every_epoch:
            self.trainer.reset_val_dataloader(model)

    def track_epoch_end_reduce_metrics(self, epoch_output, epoch_end_outputs):
        # track the outputs to reduce at the end of the epoch
        for opt_idx, opt_outputs in enumerate(epoch_end_outputs):
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
        is_result_obj = isinstance(training_step_output, Result)

        if is_result_obj:
            training_step_output.detach()
        else:
            training_step_output.batch_loss = training_step_output.batch_loss.detach()

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
        model = self.trainer.get_model()
        model._results = Result()
        model._current_fx_name = "training_step"

        with self.trainer.profiler.profile("model_forward"):
            args = self.build_train_args(split_batch, batch_idx, opt_idx, hiddens)
            training_step_output = self.trainer.accelerator_backend.training_step(args)
            self._check_training_step_output(training_step_output)

            training_step_output = self.trainer.call_hook("training_step_end", training_step_output)

            training_step_output_for_epoch_end, training_step_output = self._process_training_step_output(
                training_step_output, split_batch
            )
            is_result_obj = isinstance(training_step_output, Result)

            if training_step_output_for_epoch_end is None:
                return None

        # enable empty loss when using manual opt
        closure_loss = None
        untouched_loss = None

        if self.trainer.train_loop.automatic_optimization:
            # accumulate loss
            # (if accumulate_grad_batches = 1 no effect)
            if is_result_obj:
                closure_loss = training_step_output.minimize
            else:
                closure_loss = training_step_output.batch_loss

            closure_loss = closure_loss / self.trainer.accumulate_grad_batches

            # the loss will get scaled for amp. avoid any modifications to it
            untouched_loss = closure_loss.detach().clone()

        # result
        result = AttributeDict(
            closure_loss=closure_loss,
            loss=untouched_loss,
            training_step_output=training_step_output,
            training_step_output_for_epoch_end=training_step_output_for_epoch_end,
            hiddens=training_step_output.hiddens,
        )
        return result

    def _process_training_step_output(self, training_step_output, split_batch):
        training_step_output_for_epoch_end = training_step_output

        # enable validation_step return None
        if training_step_output_for_epoch_end is None:
            return None, None

        # -----------------------------------------
        # process result return (DEPRECATE in 1.0)
        # -----------------------------------------
        if isinstance(training_step_output, Result):
            training_step_output_for_epoch_end = self._process_result(training_step_output, split_batch)
            return training_step_output_for_epoch_end, training_step_output

        # -----------------------------------------
        # process hybrid (1.0)
        # -----------------------------------------
        # no need for these checks in 1.0.0
        # TODO: remove checks in 1.0.0
        is_tensor = isinstance(training_step_output_for_epoch_end, torch.Tensor)
        is_1_0_output = is_tensor or ("log" not in training_step_output and "progress_bar" not in training_step_output)
        if is_1_0_output:
            return self._process_training_step_output_1_0(training_step_output, split_batch)

        # -----------------------------------------
        # process old dict (deprecate 1.0)
        # -----------------------------------------
        training_step_output = self.trainer.process_dict_result(training_step_output, train=True)

        training_step_output = AttributeDict(
            batch_loss=training_step_output[0],
            pbar_on_batch_end=training_step_output[1],
            log_metrics=training_step_output[2],
            callback_metrics=training_step_output[3],
            hiddens=training_step_output[4],
        )
        # if the user decides to finally reduce things in epoch_end, save raw output without graphs
        if isinstance(training_step_output_for_epoch_end, torch.Tensor):
            training_step_output_for_epoch_end = training_step_output_for_epoch_end.detach()
        else:
            training_step_output_for_epoch_end = recursive_detach(training_step_output_for_epoch_end)

        return training_step_output_for_epoch_end, training_step_output

    def _process_training_step_output_1_0(self, training_step_output, split_batch):
        result = self.trainer.get_model()._results

        loss = None
        hiddens = None

        # handle dict return
        if isinstance(training_step_output, dict):
            loss = training_step_output.pop("loss", None)
            hiddens = training_step_output.pop("hiddens", None)
            result["extra"] = training_step_output

        # handle scalar return
        elif isinstance(training_step_output, torch.Tensor):
            loss = training_step_output
            result["extra"] = {}

        # map to results under the hood
        result.minimize = loss
        result.hiddens = hiddens

        # track batch for manual reduction with result
        result.track_batch_size(len(split_batch))

        # track metrics without grads for epoch reduction
        training_step_output_for_epoch_end = copy(result)
        training_step_output_for_epoch_end.detach()

        # what flows back into the system
        training_step_output = result

        return training_step_output_for_epoch_end, training_step_output

    def _process_result(self, training_step_output, split_batch):
        training_step_output.track_batch_size(len(split_batch))
        m = """
        TrainResult and EvalResult were deprecated in 0.9.1 and support will drop in 1.0.0.
        Use self.log and .write from the LightningModule to log metrics and write predictions.
        training_step can now only return a scalar (for the loss) or a dictionary with anything you want.

        Option 1:
        return loss

        Option 2:
        return {'loss': loss, 'anything_else': ...}

        Option 3:
        return {'loss': loss, 'hiddens': hiddens, 'anything_else': ...}
            """
        rank_zero_warn(m)

        # don't allow EvalResult in the training_step
        if isinstance(training_step_output, EvalResult):
            raise MisconfigurationException(
                "training_step cannot return EvalResult, " "use a dict or TrainResult instead"
            )

        training_step_output_for_epoch_end = copy(training_step_output)
        training_step_output_for_epoch_end.detach()

        return training_step_output_for_epoch_end

    def optimizer_step(self, optimizer, opt_idx, batch_idx, train_step_and_backward_closure):
        with self.trainer.profiler.profile("optimizer_step"):
            # optimizer step lightningModule hook
            self.trainer.accelerator_backend.optimizer_step(
                optimizer, batch_idx, opt_idx, train_step_and_backward_closure
            )

    def on_before_zero_grad(self, optimizer):
        model = self.trainer.get_model()
        model.on_before_zero_grad(optimizer)

    def optimizer_zero_grad(self, batch_idx, optimizer, opt_idx):
        self.trainer.accelerator_backend.optimizer_zero_grad(batch_idx, optimizer, opt_idx)

    def track_and_norm_grad(self, optimizer):
        # track gradient norms
        grad_norm_dic = self._track_gradient_norm()

        # clip gradients
        self.trainer.accelerator_backend.clip_gradients(optimizer)
        self._cur_grad_norm_dict = grad_norm_dic

    def _track_gradient_norm(self):
        grad_norm_dict = {}
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if float(self.trainer.track_grad_norm) > 0:
                model = self.trainer.get_model()
                grad_norm_dict = model.grad_norm(self.trainer.track_grad_norm)
        return grad_norm_dict

    def log_training_step_metrics(self, opt_closure_result, batch_callback_metrics, batch_log_metrics):
        # track callback metrics
        callback_metrics = opt_closure_result.training_step_output.callback_metrics

        # decide which metrics to log (results vs dict return)
        using_results_obj = isinstance(opt_closure_result.training_step_output, Result)
        if using_results_obj:
            metrics_to_log = opt_closure_result.training_step_output.get_batch_log_metrics(
                include_forked_originals=False
            )
            step_pbar_metrics = opt_closure_result.training_step_output.get_batch_pbar_metrics(
                include_forked_originals=False
            )
            forked_metrics = opt_closure_result.training_step_output.get_forked_metrics()
            callback_metrics.update(forked_metrics)
        else:
            metrics_to_log = opt_closure_result.training_step_output.log_metrics
            step_pbar_metrics = opt_closure_result.training_step_output.pbar_on_batch_end

        # track batch log metrics
        batch_log_metrics.append(metrics_to_log)

        # track progress bar metrics
        if len(step_pbar_metrics) > 0:
            self.trainer.logger_connector.add_progress_bar_metrics(step_pbar_metrics)
            self.trainer.logger_connector.callback_metrics.update(step_pbar_metrics)

        batch_callback_metrics.append(callback_metrics)

    def process_hiddens(self, opt_closure_result):
        hiddens = opt_closure_result.hiddens
        if isinstance(opt_closure_result.training_step_output, Result):
            opt_closure_result.training_step_output_for_epoch_end.drop_hiddens()
        return hiddens

    def tbptt_split_batch(self, batch):
        splits = [batch]
        if self.trainer.truncated_bptt_steps is not None:
            model_ref = self.trainer.get_model()
            with self.trainer.profiler.profile("tbptt_split_batch"):
                splits = model_ref.tbptt_split_batch(batch, self.trainer.truncated_bptt_steps)
        return splits

    def run_training_epoch(self):

        # get model
        model = self.trainer.get_model()

        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.trainer.accelerator_backend.process_dataloader(self.trainer.train_dataloader)

        # track epoch output
        epoch_output = [[] for _ in range(self.num_optimizers)]

        # enable profiling for the dataloader
        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(train_dataloader)
        dataloader_idx = 0
        should_check_val = False
        for batch_idx, (batch, is_last_batch) in train_dataloader:

            self.trainer.batch_idx = batch_idx

            # ------------------------------------
            # TRAINING_STEP + TRAINING_STEP_END
            # ------------------------------------
            batch_output = self.run_training_batch(batch, batch_idx, dataloader_idx)

            # when returning -1 from train_step, we end epoch early
            if batch_output.signal == -1:
                break

            # only track outputs when user implements training_epoch_end
            # otherwise we will build up unnecessary memory
            epoch_end_outputs = self.process_train_step_outputs(
                batch_output.training_step_output_for_epoch_end,
                self.early_stopping_accumulator,
                self.checkpoint_accumulator,
            )

            # hook
            # TODO: add outputs to batches
            self.on_train_batch_end(epoch_output, epoch_end_outputs, batch, batch_idx, dataloader_idx)

            # -----------------------------------------
            # SAVE METRICS TO LOGGERS
            # -----------------------------------------
            self.trainer.logger_connector.log_train_step_metrics(batch_output)

            # -----------------------------------------
            # VALIDATE IF NEEDED + CHECKPOINT CALLBACK
            # -----------------------------------------
            should_check_val = self.should_check_val_fx(batch_idx, is_last_batch)
            if should_check_val:
                self.trainer.run_evaluation(test_mode=False)

            # -----------------------------------------
            # SAVE LOGGERS (ie: Tensorboard, etc...)
            # -----------------------------------------
            self.save_loggers_on_train_batch_end()

            # update LR schedulers
            monitor_metrics = deepcopy(self.trainer.logger_connector.callback_metrics)
            monitor_metrics.update(batch_output.batch_log_metrics)
            self.update_train_loop_lr_schedulers(monitor_metrics=monitor_metrics)
            self.trainer.checkpoint_connector.has_trained = True

            # max steps reached, end training
            if self.trainer.max_steps is not None and self.trainer.max_steps == self.trainer.global_step + 1:
                accumulation_done = self._accumulated_batches_reached()
                # Ensure accumulation across batches has completed before breaking loop
                if accumulation_done:
                    break

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if self.trainer.should_stop:
                break

            self.trainer.total_batch_idx += 1

            # stop epoch if we limited the number of training batches
            if (batch_idx + 1) >= self.trainer.num_training_batches:
                break

            # progress global step according to grads progress
            self.increment_accumulated_grad_global_step()

        # log epoch metrics
        self.trainer.logger_connector.log_train_epoch_end_metrics(
            epoch_output, self.checkpoint_accumulator, self.early_stopping_accumulator, self.num_optimizers
        )

        # hook
        self.trainer.logger_connector.on_train_epoch_end(epoch_output)

        # when no val loop is present or fast-dev-run still need to call checkpoints
        self.check_checkpoint_callback(not (should_check_val or is_overridden("validation_step", model)))

        # epoch end hook
        self.run_on_epoch_end_hook(epoch_output)

        # increment the global step once
        # progress global step according to grads progress
        self.increment_accumulated_grad_global_step()

    def run_training_batch(self, batch, batch_idx, dataloader_idx):
        # track grad norms
        grad_norm_dic = {}

        # track all metrics for callbacks
        batch_callback_metrics = []

        # track metrics to log
        batch_log_metrics = []

        # bookkeeping
        using_results_obj = False
        self.trainer.hiddens = None

        # track all outputs across time and num of optimizers
        batch_outputs = [[] for _ in range(len(self.get_optimizers_iterable()))]

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
            self.trainer.split_idx = split_idx

            # in manual optimization we loop over all optimizers at once
            optimizers = self.get_optimizers_iterable()
            if not self.automatic_optimization:
                optimizers = [optimizers[0]]

            # loop over optimizers
            for opt_idx, optimizer in optimizers:
                # make sure only the gradients of the current optimizer's parameters are calculated
                # in the training step to prevent dangling gradients in multiple-optimizer setup.
                if self.automatic_optimization and len(self.trainer.optimizers) > 1:
                    model = self.trainer.get_model()
                    model.toggle_optimizer(optimizer, opt_idx)

                if self.should_accumulate():
                    # For gradient accumulation

                    # -------------------
                    # calculate loss (train step + train step end)
                    # -------------------

                    # perform dpp sync only when performing optimizer_step
                    with self.block_ddp_sync_behaviour():
                        self.training_step_and_backward(split_batch, batch_idx, opt_idx, optimizer, self.trainer.hiddens)

                    batch_outputs = self._process_closure_result(
                        batch_callback_metrics=batch_callback_metrics,
                        batch_log_metrics=batch_log_metrics,
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
                                split_batch,
                                batch_idx,
                                opt_idx,
                                optimizer,
                                self.trainer.hiddens
                            )
                            return None if result is None else result.loss

                        # optimizer step
                        self.optimizer_step(optimizer, opt_idx, batch_idx, train_step_and_backward_closure)

                    else:
                        self._curr_step_result = self.training_step(split_batch, batch_idx, opt_idx, self.trainer.hiddens)

                    if self._curr_step_result is None:
                        # user decided to skip optimization
                        # make sure to zero grad.
                        self.zero_grad_handler(batch_idx, optimizer, opt_idx)
                        continue

                    batch_outputs = self._process_closure_result(
                        batch_callback_metrics=batch_callback_metrics,
                        batch_log_metrics=batch_log_metrics,
                        batch_outputs=batch_outputs,
                        opt_idx=opt_idx,
                    )

                    grad_norm_dic = self._cur_grad_norm_dict
                    self._cur_grad_norm_dict = None

                    # hook + clear gradients
                    self.zero_grad_handler(batch_idx, optimizer, opt_idx)

                    # update running loss + reset accumulated loss
                    self.update_running_loss()

        # collapse all metrics into one dict
        batch_log_metrics = {k: v for d in batch_log_metrics for k, v in d.items()}

        # track all metrics for callbacks
        self.trainer.logger_connector.callback_metrics.update(batch_log_metrics)
        self.trainer.logger_connector.callback_metrics.update(
            {k: v for d in batch_callback_metrics for k, v in d.items() if v is not None}
        )

        result = AttributeDict(
            signal=0,
            grad_norm_dic=grad_norm_dic,
            batch_log_metrics=batch_log_metrics,
            training_step_output_for_epoch_end=batch_outputs,
        )
        return result

    @contextmanager
    def block_ddp_sync_behaviour(self):
        if isinstance(self.trainer.model, torch.nn.parallel.DistributedDataParallel):
            yield self.trainer.model.no_sync()
        else:
            yield

    def _process_closure_result(
        self, batch_callback_metrics: list, batch_log_metrics: list, batch_outputs: list, opt_idx: int
    ) -> list:
        opt_closure_result = self._curr_step_result

        if opt_closure_result is not None:

            # log metrics
            self.log_training_step_metrics(opt_closure_result, batch_callback_metrics, batch_log_metrics)

            # track hiddens
            self.trainer.hiddens = self.process_hiddens(opt_closure_result)

            # check if loss or model weights are nan
            if self.trainer.terminate_on_nan:
                self.trainer.detect_nan_tensors(opt_closure_result.loss)

            # track all the outputs across all steps
            batch_opt_idx = opt_idx if len(batch_outputs) > 1 else 0
            batch_outputs[batch_opt_idx].append(opt_closure_result.training_step_output_for_epoch_end)

            if self.automatic_optimization:
                # track total loss for logging (avoid mem leaks)
                self.accumulated_loss.append(opt_closure_result.loss)

        self._curr_step_result = None

        return batch_outputs

    def training_step_and_backward(self, split_batch, batch_idx, opt_idx, optimizer, hiddens):
        """
        wrap the forward step in a closure so second order methods work
        """
        # lightning module hook
        result = self.training_step(split_batch, batch_idx, opt_idx, hiddens)
        self._curr_step_result = result

        if result is None:
            self.warning_cache.warn("training_step returned None if it was on purpose, ignore this warning...")
            return None

        if self.trainer.train_loop.automatic_optimization:
            # backward pass
            with self.trainer.profiler.profile("model_backward"):
                self.backward(result, optimizer, opt_idx)

            # hook - call this hook only
            # when gradients have finished to accumulate
            if not self.should_accumulate():
                self.on_after_backward(result.training_step_output, batch_idx, result.loss)

            # check if loss or model weights are nan
            if self.trainer.terminate_on_nan:
                self.trainer.detect_nan_tensors(result.loss)

        return result

    def backward(self, result, optimizer, opt_idx, *args, **kwargs):
        self.trainer.dev_debugger.track_event("backward_call")

        # backward can be called manually in the training loop
        if isinstance(result, torch.Tensor):
            self.trainer.accelerator_backend.backward(result, optimizer, opt_idx, *args, **kwargs)
        else:
            result.closure_loss = self.trainer.accelerator_backend.backward(
                result.closure_loss, optimizer, opt_idx, *args, **kwargs
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

    def run_on_epoch_end_hook(self, epoch_output):
        self.trainer.call_hook("on_epoch_end")
        self.trainer.call_hook("on_train_epoch_end", epoch_output)

    def increment_accumulated_grad_global_step(self):
        num_accumulated_batches_reached = self._accumulated_batches_reached()
        num_training_batches_reached = self._num_training_batches_reached()

        # progress global step according to grads progress
        if num_accumulated_batches_reached or num_training_batches_reached:
            self.trainer.global_step += 1

    def _accumulated_batches_reached(self):
        return (self.trainer.batch_idx + 1) % self.trainer.accumulate_grad_batches == 0

    def _num_training_batches_reached(self):
        return (self.trainer.batch_idx + 1) == self.trainer.num_training_batches

    def should_accumulate(self):
        # checks if backward or backward + optimizer step (via closure)
        accumulation_done = self._accumulated_batches_reached()
        is_final_batch = self._num_training_batches_reached()
        return not (accumulation_done or is_final_batch)

    def should_check_val_fx(self, batch_idx, is_last_batch):
        # decide if we should run validation
        is_val_check_batch = (batch_idx + 1) % self.trainer.val_check_batch == 0
        is_val_check_epoch = (self.trainer.current_epoch + 1) % self.trainer.check_val_every_n_epoch == 0
        can_check_val = self.trainer.enable_validation and is_val_check_epoch
        should_check_val = is_val_check_batch or self.trainer.should_stop
        is_last_batch_for_infinite_dataset = is_last_batch and self.trainer.val_check_batch == float("inf")
        should_check_val = can_check_val and (should_check_val or is_last_batch_for_infinite_dataset)

        return should_check_val

    def build_train_args(self, batch, batch_idx, opt_idx, hiddens):
        # enable not needing to add opt_idx to training_step
        args = [batch, batch_idx]

        if len(self.trainer.optimizers) > 1:
            if self.trainer.has_arg("training_step", "optimizer_idx"):
                args.append(opt_idx)
            else:
                num_opts = len(self.trainer.optimizers)
                raise ValueError(
                    f"Your LightningModule defines {num_opts} optimizers but "
                    f'training_step is missing the "optimizer_idx" argument.'
                )

        # pass hiddens if using tbptt
        if self.trainer.truncated_bptt_steps is not None:
            args.append(hiddens)

        return args

    def save_loggers_on_train_batch_end(self):
        # when loggers should save to disk
        should_save_log = (
            self.trainer.global_step + 1
        ) % self.trainer.flush_logs_every_n_steps == 0 or self.trainer.should_stop
        if should_save_log or self.trainer.fast_dev_run:
            if self.trainer.is_global_zero and self.trainer.logger is not None:
                self.trainer.logger.save()

    def process_train_step_outputs(self, all_train_step_outputs, early_stopping_accumulator, checkpoint_accumulator):
        """
        Figure out what needs to be tracked/logged at the end of the epoch
        """

        # the training step outputs a list per optimizer. The list contains the outputs at each time step
        # when no TBPTT is used, then the list has 1 item per batch
        # when TBPTT IS used, then the list has n items (1 per time step)
        epoch_end_outputs = []
        for optimizer_idx_outputs in all_train_step_outputs:
            # extract one representative sample from each time step (1 if no tbptt) and 0th optimizer
            if len(optimizer_idx_outputs) == 0:
                continue

            sample_output = optimizer_idx_outputs[-1]

            # pull out callback info if available (ie: Results object)
            if isinstance(sample_output, dict) and "early_stop_on" in sample_output:
                early_stopping_accumulator.accumulate(sample_output["early_stop_on"])

            if isinstance(sample_output, dict) and "checkpoint_on" in sample_output:
                checkpoint_accumulator.accumulate(sample_output["checkpoint_on"])

            # decide if we need to reduce at the end of the epoch automatically
            auto_reduce_tng_result = isinstance(sample_output, Result) and sample_output.should_reduce_on_epoch_end

            # only track when a) it needs to be autoreduced OR b) the user wants to manually reduce on epoch end
            if is_overridden("training_epoch_end", model=self.trainer.get_model()) or auto_reduce_tng_result:
                epoch_end_outputs.append(optimizer_idx_outputs)

        return epoch_end_outputs

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
            model = self.trainer.get_model()
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

    def zero_grad_handler(self, batch_idx, optimizer, opt_idx):
        if self.automatic_optimization:
            # hook
            self.on_before_zero_grad(optimizer)
            optimizers = enumerate([optimizer])
        else:
            optimizers = self.get_optimizers_iterable()

        for idx, optimizer in optimizers:
            self.optimizer_zero_grad(batch_idx, optimizer, opt_idx)
