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
import pytorch_lightning as pl
from pytorch_lightning.plugins.precision.precision_plugin import PrecisionPlugin
from pytorch_lightning.strategies import DataParallelStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.warnings import PossibleUserWarning


def verify_loop_configurations(trainer: "pl.Trainer") -> None:
    r"""
    Checks that the model is configured correctly before the run is started.

    Args:
        trainer: Lightning Trainer. Its `lightning_module` (the model) to check the configuration.

    """
    model = trainer.lightning_module

    if trainer.state.fn is None:
        raise ValueError("Unexpected: Trainer state fn must be set before validating loop configuration.")
    if trainer.state.fn in (TrainerFn.FITTING, TrainerFn.TUNING):
        __verify_train_val_loop_configuration(trainer, model)
        __verify_manual_optimization_support(trainer, model)
        __check_training_step_requires_dataloader_iter(model)
    elif trainer.state.fn == TrainerFn.VALIDATING:
        __verify_eval_loop_configuration(trainer, model, "val")
    elif trainer.state.fn == TrainerFn.TESTING:
        __verify_eval_loop_configuration(trainer, model, "test")
    elif trainer.state.fn == TrainerFn.PREDICTING:
        __verify_eval_loop_configuration(trainer, model, "predict")

    __verify_dp_batch_transfer_support(trainer, model)
    _check_deprecated_callback_hooks(trainer)
    # TODO: Delete _check_on_hpc_hooks in v1.8
    _check_on_hpc_hooks(model)
    # TODO: Delete on_epoch_start/on_epoch_end hooks in v1.8
    _check_on_epoch_start_end(model)
    # TODO: Delete CheckpointHooks off PrecisionPlugin in v1.8
    _check_precision_plugin_checkpoint_hooks(trainer)
    # TODO: Delete on_pretrain_routine_start/end hooks in v1.8
    _check_on_pretrain_routine(model)
    # TODO: Delete CheckpointHooks off LightningDataModule in v1.8
    _check_datamodule_checkpoint_hooks(trainer)
    _check_setup_method(trainer)


def __verify_train_val_loop_configuration(trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
    # -----------------------------------
    # verify model has a training step
    # -----------------------------------
    has_training_step = is_overridden("training_step", model)
    if not has_training_step:
        raise MisconfigurationException(
            "No `training_step()` method defined. Lightning `Trainer` expects as minimum a"
            " `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined."
        )

    # -----------------------------------
    # verify model has optimizer
    # -----------------------------------
    has_optimizers = is_overridden("configure_optimizers", model)
    if not has_optimizers:
        raise MisconfigurationException(
            "No `configure_optimizers()` method defined. Lightning `Trainer` expects as minimum a"
            " `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined."
        )

    trainer.overridden_optimizer_step = is_overridden("optimizer_step", model)
    trainer.overridden_optimizer_zero_grad = is_overridden("optimizer_zero_grad", model)
    automatic_optimization = model.automatic_optimization
    going_to_accumulate_grad_batches = trainer.accumulation_scheduler.going_to_accumulate_grad_batches()

    has_overridden_optimization_functions = trainer.overridden_optimizer_step or trainer.overridden_optimizer_zero_grad
    if has_overridden_optimization_functions and going_to_accumulate_grad_batches and automatic_optimization:
        rank_zero_warn(
            "When using `Trainer(accumulate_grad_batches != 1)` and overriding"
            " `LightningModule.optimizer_{step,zero_grad}`, the hooks will not be called on every batch"
            " (rather, they are called on every optimization step)."
        )

    # -----------------------------------
    # verify model for val loop
    # -----------------------------------

    has_val_loader = trainer._data_connector._val_dataloader_source.is_defined()
    has_val_step = is_overridden("validation_step", model)

    if has_val_loader and not has_val_step:
        rank_zero_warn("You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.")
    if has_val_step and not has_val_loader:
        rank_zero_warn(
            "You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.",
            category=PossibleUserWarning,
        )


def __verify_eval_loop_configuration(trainer: "pl.Trainer", model: "pl.LightningModule", stage: str) -> None:
    step_name = "validation_step" if stage == "val" else f"{stage}_step"
    trainer_method = "validate" if stage == "val" else stage

    has_step = is_overridden(step_name, model)

    # predict_step is not required to be overridden
    if stage == "predict":
        if model.predict_step is None:
            raise MisconfigurationException("`predict_step` cannot be None to run `Trainer.predict`")
        elif not has_step and not is_overridden("forward", model):
            raise MisconfigurationException("`Trainer.predict` requires `forward` method to run.")
    else:
        # -----------------------------------
        # verify model has an eval_step
        # -----------------------------------
        if not has_step:
            raise MisconfigurationException(f"No `{step_name}()` method defined to run `Trainer.{trainer_method}`.")


def __verify_dp_batch_transfer_support(trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
    """Raise Misconfiguration exception since these hooks are not supported in DP mode."""
    # TODO: Remove this blocker once batch transfer to device is integrated in Lightning for DP mode.
    batch_transfer_hooks = ("on_before_batch_transfer", "transfer_batch_to_device", "on_after_batch_transfer")
    datahook_selector = trainer._data_connector._datahook_selector
    for hook in batch_transfer_hooks:
        if isinstance(trainer.strategy, DataParallelStrategy) and (
            is_overridden(hook, datahook_selector.model) or is_overridden(hook, datahook_selector.datamodule)
        ):
            raise MisconfigurationException(f"Overriding `{hook}` is not supported in DP mode.")


def __verify_manual_optimization_support(trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
    if model.automatic_optimization:
        return
    if trainer.gradient_clip_val is not None and trainer.gradient_clip_val > 0:
        raise MisconfigurationException(
            "Automatic gradient clipping is not supported for manual optimization."
            f" Remove `Trainer(gradient_clip_val={trainer.gradient_clip_val})`"
            " or switch to automatic optimization."
        )
    if trainer.accumulate_grad_batches != 1:
        raise MisconfigurationException(
            "Automatic gradient accumulation is not supported for manual optimization."
            f" Remove `Trainer(accumulate_grad_batches={trainer.accumulate_grad_batches})`"
            " or switch to automatic optimization."
        )


def __check_training_step_requires_dataloader_iter(model: "pl.LightningModule") -> None:
    """Check if the current `training_step` is requesting `dataloader_iter`."""
    training_step_fx = model.training_step
    if is_param_in_hook_signature(training_step_fx, "dataloader_iter", explicit=True):

        if is_overridden("on_train_batch_start", model):
            raise MisconfigurationException(
                "The model hook `on_train_batch_start` is not compatible with "
                "taking a `dataloader_iter` argument in your `training_step`."
            )

        if is_overridden("on_train_batch_end", model):
            raise MisconfigurationException(
                "The model hook `on_train_batch_end` is not compatible with "
                "taking a `dataloader_iter` argument in your `training_step`."
            )

        if model.truncated_bptt_steps > 0:
            raise MisconfigurationException(
                "The model taking a `dataloader_iter` argument in your `training_step` "
                "is incompatible with `truncated_bptt_steps > 0`."
            )


# TODO: Delete _check_on_hpc_hooks in v1.8
def _check_on_hpc_hooks(model: "pl.LightningModule") -> None:
    if is_overridden("on_hpc_save", model):
        rank_zero_deprecation(
            "Method `LightningModule.on_hpc_save` is deprecated in v1.6 and"
            " will be removed in v1.8. Please use `LightningModule.on_save_checkpoint` instead."
        )

    if is_overridden("on_hpc_load", model):
        rank_zero_deprecation(
            "Method `LightningModule.on_hpc_load` is deprecated in v1.6 and"
            " will be removed in v1.8. Please use `LightningModule.on_load_checkpoint` instead."
        )


# TODO: Remove on_epoch_start/on_epoch_end hooks in v1.8
def _check_on_epoch_start_end(model: "pl.LightningModule") -> None:
    hooks = (
        ("on_epoch_start", "on_<train/validation/test>_epoch_start"),
        ("on_epoch_end", "on_<train/validation/test>_epoch_end"),
    )

    for hook, alternative_hook in hooks:
        if is_overridden(hook, model):
            rank_zero_deprecation(
                f"The `LightningModule.{hook}` hook was deprecated in v1.6 and"
                f" will be removed in v1.8. Please use `LightningModule.{alternative_hook}` instead."
            )


def _check_on_pretrain_routine(model: "pl.LightningModule") -> None:
    hooks = (("on_pretrain_routine_start", "on_fit_start"), ("on_pretrain_routine_end", "on_fit_start"))
    for hook, alternative_hook in hooks:
        if is_overridden(hook, model):
            rank_zero_deprecation(
                f"The `LightningModule.{hook}` hook was deprecated in v1.6 and"
                f" will be removed in v1.8. Please use `LightningModule.{alternative_hook}` instead."
            )


def _check_deprecated_callback_hooks(trainer: "pl.Trainer") -> None:
    for callback in trainer.callbacks:
        if is_overridden(method_name="on_init_start", instance=callback):
            rank_zero_deprecation(
                "The `on_init_start` callback hook was deprecated in v1.6 and will be removed in v1.8."
            )
        if is_overridden(method_name="on_init_end", instance=callback):
            rank_zero_deprecation("The `on_init_end` callback hook was deprecated in v1.6 and will be removed in v1.8.")

        if is_overridden(method_name="on_configure_sharded_model", instance=callback):
            rank_zero_deprecation(
                "The `on_configure_sharded_model` callback hook was deprecated in"
                " v1.6 and will be removed in v1.8. Use `setup()` instead."
            )
        if is_overridden(method_name="on_before_accelerator_backend_setup", instance=callback):
            rank_zero_deprecation(
                "The `on_before_accelerator_backend_setup` callback hook was deprecated in"
                " v1.6 and will be removed in v1.8. Use `setup()` instead."
            )
        if is_overridden(method_name="on_load_checkpoint", instance=callback):
            rank_zero_deprecation(
                f"`{callback.__class__.__name__}.on_load_checkpoint` will change its signature and behavior in v1.8."
                " If you wish to load the state of the callback, use `load_state_dict` instead."
                " In v1.8 `on_load_checkpoint(..., checkpoint)` will receive the entire loaded"
                " checkpoint dictionary instead of callback state."
            )

        for hook, alternative_hook in (
            ["on_batch_start", "on_train_batch_start"],
            ["on_batch_end", "on_train_batch_end"],
        ):
            if is_overridden(method_name=hook, instance=callback):
                rank_zero_deprecation(
                    f"The `Callback.{hook}` hook was deprecated in v1.6 and"
                    f" will be removed in v1.8. Please use `Callback.{alternative_hook}` instead."
                )
        for hook, alternative_hook in (
            ["on_epoch_start", "on_<train/validation/test>_epoch_start"],
            ["on_epoch_end", "on_<train/validation/test>_epoch_end"],
        ):
            if is_overridden(method_name=hook, instance=callback):
                rank_zero_deprecation(
                    f"The `Callback.{hook}` hook was deprecated in v1.6 and"
                    f" will be removed in v1.8. Please use `Callback.{alternative_hook}` instead."
                )
        for hook in ("on_pretrain_routine_start", "on_pretrain_routine_end"):
            if is_overridden(method_name=hook, instance=callback):
                rank_zero_deprecation(
                    f"The `Callback.{hook}` hook has been deprecated in v1.6 and"
                    " will be removed in v1.8. Please use `Callback.on_fit_start` instead."
                )


def _check_precision_plugin_checkpoint_hooks(trainer: "pl.Trainer") -> None:
    if is_overridden(method_name="on_save_checkpoint", instance=trainer.precision_plugin, parent=PrecisionPlugin):
        rank_zero_deprecation(
            "`PrecisionPlugin.on_save_checkpoint` was deprecated in"
            " v1.6 and will be removed in v1.8. Use `state_dict` instead."
        )
    if is_overridden(method_name="on_load_checkpoint", instance=trainer.precision_plugin, parent=PrecisionPlugin):
        rank_zero_deprecation(
            "`PrecisionPlugin.on_load_checkpoint` was deprecated in"
            " v1.6 and will be removed in v1.8. Use `load_state_dict` instead."
        )


def _check_datamodule_checkpoint_hooks(trainer: "pl.Trainer") -> None:
    if is_overridden(method_name="on_save_checkpoint", instance=trainer.datamodule):
        rank_zero_deprecation(
            "`LightningDataModule.on_save_checkpoint` was deprecated in"
            " v1.6 and will be removed in v1.8. Use `state_dict` instead."
        )
    if is_overridden(method_name="on_load_checkpoint", instance=trainer.datamodule):
        rank_zero_deprecation(
            "`LightningDataModule.on_load_checkpoint` was deprecated in"
            " v1.6 and will be removed in v1.8. Use `load_state_dict` instead."
        )


def _check_setup_method(trainer: "pl.Trainer") -> None:
    for obj in [trainer.lightning_module, trainer.datamodule] + trainer.callbacks:
        if is_overridden("setup", obj) and not is_param_in_hook_signature(obj.setup, "stage"):
            raise MisconfigurationException(f"`{obj.__class__.__name__}.setup` does not have a `stage` argument.")
