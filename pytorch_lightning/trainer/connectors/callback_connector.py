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
import os
from datetime import timedelta
from typing import Dict, List, Optional, Sequence, Union

from pytorch_lightning.callbacks import (
    Callback,
    GradientAccumulationScheduler,
    ModelCheckpoint,
    ModelSummary,
    ProgressBarBase,
    RichProgressBar,
    TQDMProgressBar,
)
from pytorch_lightning.callbacks.rich_model_summary import RichModelSummary
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.utilities.enums import ModelSummaryMode
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_info


class CallbackConnector:
    def __init__(self, trainer):
        self.trainer = trainer

    def on_trainer_init(
        self,
        callbacks: Optional[Union[List[Callback], Callback]],
        checkpoint_callback: Optional[bool],
        enable_checkpointing: bool,
        enable_progress_bar: bool,
        progress_bar_refresh_rate: Optional[int],
        process_position: int,
        default_root_dir: Optional[str],
        weights_save_path: Optional[str],
        enable_model_summary: bool,
        weights_summary: Optional[str],
        stochastic_weight_avg: bool,
        max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None,
        accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None,
    ):
        # init folder paths for checkpoint + weights save callbacks
        self.trainer._default_root_dir = default_root_dir or os.getcwd()
        if weights_save_path:
            rank_zero_deprecation(
                "Setting `Trainer(weights_save_path=)` has been deprecated in v1.6 and will be"
                " removed in v1.8. Please pass ``dirpath`` directly to the `ModelCheckpoint` callback"
            )

        self.trainer._weights_save_path = weights_save_path or self.trainer._default_root_dir
        if stochastic_weight_avg:
            rank_zero_deprecation(
                "Setting `Trainer(stochastic_weight_avg=True)` is deprecated in v1.5 and will be removed in v1.7."
                " Please pass `pytorch_lightning.callbacks.stochastic_weight_avg.StochasticWeightAveraging`"
                " directly to the Trainer's `callbacks` argument instead."
            )
        self.trainer._stochastic_weight_avg = stochastic_weight_avg

        # init callbacks
        if isinstance(callbacks, Callback):
            callbacks = [callbacks]
        self.trainer.callbacks = callbacks or []

        # configure checkpoint callback
        # pass through the required args to figure out defaults
        self._configure_checkpoint_callbacks(checkpoint_callback, enable_checkpointing)

        # configure swa callback
        self._configure_swa_callbacks()

        # configure the timer callback.
        # responsible to stop the training when max_time is reached.
        self._configure_timer_callback(max_time)

        # init progress bar
        if process_position != 0:
            rank_zero_deprecation(
                f"Setting `Trainer(process_position={process_position})` is deprecated in v1.5 and will be removed"
                " in v1.7. Please pass `pytorch_lightning.callbacks.progress.TQDMProgressBar` with"
                " `process_position` directly to the Trainer's `callbacks` argument instead."
            )

        if progress_bar_refresh_rate is not None:
            rank_zero_deprecation(
                f"Setting `Trainer(progress_bar_refresh_rate={progress_bar_refresh_rate})` is deprecated in v1.5 and"
                " will be removed in v1.7. Please pass `pytorch_lightning.callbacks.progress.TQDMProgressBar` with"
                " `refresh_rate` directly to the Trainer's `callbacks` argument instead. Or, to disable the progress"
                " bar pass `enable_progress_bar = False` to the Trainer."
            )

        self._configure_progress_bar(progress_bar_refresh_rate, process_position, enable_progress_bar)

        # configure the ModelSummary callback
        self._configure_model_summary_callback(enable_model_summary, weights_summary)

        # accumulated grads
        self._configure_accumulated_gradients(accumulate_grad_batches)

        if self.trainer.state._fault_tolerant_mode.is_enabled:
            self._configure_fault_tolerance_callbacks()

        # push all model checkpoint callbacks to the end
        # it is important that these are the last callbacks to run
        self.trainer.callbacks = self._reorder_callbacks(self.trainer.callbacks)

    def _configure_accumulated_gradients(
        self, accumulate_grad_batches: Optional[Union[int, Dict[int, int]]] = None
    ) -> None:
        grad_accum_callback = [cb for cb in self.trainer.callbacks if isinstance(cb, GradientAccumulationScheduler)]

        if grad_accum_callback:
            if accumulate_grad_batches is not None:
                raise MisconfigurationException(
                    "You have set both `accumulate_grad_batches` and passed an instance of "
                    "`GradientAccumulationScheduler` inside callbacks. Either remove `accumulate_grad_batches` "
                    "from trainer or remove `GradientAccumulationScheduler` from callbacks list."
                )
            grad_accum_callback = grad_accum_callback[0]
        else:
            if accumulate_grad_batches is None:
                accumulate_grad_batches = 1

            if isinstance(accumulate_grad_batches, dict):
                grad_accum_callback = GradientAccumulationScheduler(accumulate_grad_batches)
            elif isinstance(accumulate_grad_batches, int):
                grad_accum_callback = GradientAccumulationScheduler({0: accumulate_grad_batches})
            else:
                raise MisconfigurationException(
                    f"`accumulate_grad_batches` should be an int or a dict. Got {accumulate_grad_batches}."
                )

            self.trainer.callbacks.append(grad_accum_callback)

        self.trainer.accumulate_grad_batches = grad_accum_callback.get_accumulate_grad_batches(0)
        self.trainer.accumulation_scheduler = grad_accum_callback

    def _configure_checkpoint_callbacks(self, checkpoint_callback: Optional[bool], enable_checkpointing: bool) -> None:
        if checkpoint_callback is not None:
            rank_zero_deprecation(
                f"Setting `Trainer(checkpoint_callback={checkpoint_callback})` is deprecated in v1.5 and will "
                f"be removed in v1.7. Please consider using `Trainer(enable_checkpointing={checkpoint_callback})`."
            )
            # if both are set then checkpoint only if both are True
            enable_checkpointing = checkpoint_callback and enable_checkpointing

        if self.trainer.checkpoint_callbacks:
            if not enable_checkpointing:
                raise MisconfigurationException(
                    "Trainer was configured with `enable_checkpointing=False`"
                    " but found `ModelCheckpoint` in callbacks list."
                )
        elif enable_checkpointing:
            self.trainer.callbacks.append(ModelCheckpoint())

    def _configure_model_summary_callback(
        self, enable_model_summary: bool, weights_summary: Optional[str] = None
    ) -> None:
        if weights_summary is None:
            rank_zero_deprecation(
                "Setting `Trainer(weights_summary=None)` is deprecated in v1.5 and will be removed"
                " in v1.7. Please set `Trainer(enable_model_summary=False)` instead."
            )
            return
        if not enable_model_summary:
            return

        model_summary_cbs = [type(cb) for cb in self.trainer.callbacks if isinstance(cb, ModelSummary)]
        if model_summary_cbs:
            rank_zero_info(
                f"Trainer already configured with model summary callbacks: {model_summary_cbs}."
                " Skipping setting a default `ModelSummary` callback."
            )
            return

        if weights_summary == "top":
            # special case the default value for weights_summary to preserve backward compatibility
            max_depth = 1
        else:
            rank_zero_deprecation(
                f"Setting `Trainer(weights_summary={weights_summary})` is deprecated in v1.5 and will be removed"
                " in v1.7. Please pass `pytorch_lightning.callbacks.model_summary.ModelSummary` with"
                " `max_depth` directly to the Trainer's `callbacks` argument instead."
            )
            if weights_summary not in ModelSummaryMode.supported_types():
                raise MisconfigurationException(
                    f"`weights_summary` can be None, {', '.join(ModelSummaryMode.supported_types())}",
                    f" but got {weights_summary}",
                )
            max_depth = ModelSummaryMode.get_max_depth(weights_summary)

        progress_bar_callback = self.trainer.progress_bar_callback
        is_progress_bar_rich = isinstance(progress_bar_callback, RichProgressBar)

        if progress_bar_callback is not None and is_progress_bar_rich:
            model_summary = RichModelSummary(max_depth=max_depth)
        else:
            model_summary = ModelSummary(max_depth=max_depth)
        self.trainer.callbacks.append(model_summary)
        self.trainer._weights_summary = weights_summary

    def _configure_swa_callbacks(self):
        if not self.trainer._stochastic_weight_avg:
            return

        from pytorch_lightning.callbacks.stochastic_weight_avg import StochasticWeightAveraging

        existing_swa = [cb for cb in self.trainer.callbacks if isinstance(cb, StochasticWeightAveraging)]
        if not existing_swa:
            self.trainer.callbacks = [StochasticWeightAveraging()] + self.trainer.callbacks

    def _configure_progress_bar(
        self, refresh_rate: Optional[int] = None, process_position: int = 0, enable_progress_bar: bool = True
    ) -> None:
        progress_bars = [c for c in self.trainer.callbacks if isinstance(c, ProgressBarBase)]
        if len(progress_bars) > 1:
            raise MisconfigurationException(
                "You added multiple progress bar callbacks to the Trainer, but currently only one"
                " progress bar is supported."
            )
        if len(progress_bars) == 1:
            # the user specified the progress bar in the callbacks list
            # so the trainer doesn't need to provide a default one
            if enable_progress_bar:
                return

            # otherwise the user specified a progress bar callback but also
            # elected to disable the progress bar with the trainer flag
            progress_bar_callback = progress_bars[0]
            raise MisconfigurationException(
                "Trainer was configured with `enable_progress_bar=False`"
                f" but found `{progress_bar_callback.__class__.__name__}` in callbacks list."
            )

        # Return early if the user intends to disable the progress bar callback
        if refresh_rate == 0 or not enable_progress_bar:
            return
        if refresh_rate is None:
            refresh_rate = 1

        progress_bar_callback = TQDMProgressBar(refresh_rate=refresh_rate, process_position=process_position)
        self.trainer.callbacks.append(progress_bar_callback)

    def _configure_timer_callback(self, max_time: Optional[Union[str, timedelta, Dict[str, int]]] = None) -> None:
        if max_time is None:
            return
        if any(isinstance(cb, Timer) for cb in self.trainer.callbacks):
            rank_zero_info("Ignoring `Trainer(max_time=...)`, callbacks list already contains a Timer.")
            return
        timer = Timer(duration=max_time, interval="step")
        self.trainer.callbacks.append(timer)

    def _configure_fault_tolerance_callbacks(self):
        from pytorch_lightning.callbacks.fault_tolerance import _FaultToleranceCheckpoint

        if any(isinstance(cb, _FaultToleranceCheckpoint) for cb in self.trainer.callbacks):
            raise RuntimeError("There should be only one fault-tolerance checkpoint callback.")
        # don't use `log_dir` to minimize the chances of failure
        self.trainer.callbacks.append(_FaultToleranceCheckpoint(dirpath=self.trainer.default_root_dir))

    def _attach_model_logging_functions(self):
        lightning_module = self.trainer.lightning_module
        for callback in self.trainer.callbacks:
            callback.log = lightning_module.log
            callback.log_dict = lightning_module.log_dict

    def _attach_model_callbacks(self) -> None:
        """Attaches the callbacks defined in the model.

        If a callback returned by the model's configure_callback method has the same type as one or several
        callbacks already present in the trainer callbacks list, it will replace them.
        In addition, all :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callbacks
        will be pushed to the end of the list, ensuring they run last.
        """
        model_callbacks = self.trainer._call_lightning_module_hook("configure_callbacks")
        if not model_callbacks:
            return

        model_callbacks = [model_callbacks] if not isinstance(model_callbacks, Sequence) else model_callbacks
        model_callback_types = {type(c) for c in model_callbacks}
        trainer_callback_types = {type(c) for c in self.trainer.callbacks}
        override_types = model_callback_types.intersection(trainer_callback_types)
        if override_types:
            rank_zero_info(
                "The following callbacks returned in `LightningModule.configure_callbacks` will override"
                " existing callbacks passed to Trainer:"
                f" {', '.join(sorted(t.__name__ for t in override_types))}"
            )
        # remove all callbacks with a type that occurs in model callbacks
        all_callbacks = [c for c in self.trainer.callbacks if type(c) not in override_types]
        all_callbacks.extend(model_callbacks)
        all_callbacks = CallbackConnector._reorder_callbacks(all_callbacks)
        # TODO: connectors refactor: move callbacks list to connector and do not write Trainer state
        self.trainer.callbacks = all_callbacks

    @staticmethod
    def _reorder_callbacks(callbacks: List[Callback]) -> List[Callback]:
        """Moves all ModelCheckpoint callbacks to the end of the list. The sequential order within the group of
        checkpoint callbacks is preserved, as well as the order of all other callbacks.

        Args:
            callbacks: A list of callbacks.

        Return:
            A new list in which the last elements are ModelCheckpoints if there were any present in the
            input.
        """
        checkpoints = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
        not_checkpoints = [c for c in callbacks if not isinstance(c, ModelCheckpoint)]
        return not_checkpoints + checkpoints
