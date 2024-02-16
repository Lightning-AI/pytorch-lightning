# Copyright The Lightning AI team.
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

import lightning.pytorch as pl
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.trainer.states import TrainerFn
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from lightning.pytorch.utilities.signature_utils import is_param_in_hook_signature


def _verify_loop_configurations(trainer: "pl.Trainer") -> None:
    r"""Checks that the model is configured correctly before the run is started.

    Args:
        trainer: Lightning Trainer. Its `lightning_module` (the model) to check the configuration.

    """
    model = trainer.lightning_module

    if trainer.state.fn is None:
        raise ValueError("Unexpected: Trainer state fn must be set before validating loop configuration.")
    if trainer.state.fn == TrainerFn.FITTING:
        __verify_train_val_loop_configuration(trainer, model)
        __verify_manual_optimization_support(trainer, model)
    elif trainer.state.fn == TrainerFn.VALIDATING:
        __verify_eval_loop_configuration(model, "val")
    elif trainer.state.fn == TrainerFn.TESTING:
        __verify_eval_loop_configuration(model, "test")
    elif trainer.state.fn == TrainerFn.PREDICTING:
        __verify_eval_loop_configuration(model, "predict")

    __verify_configure_model_configuration(model)
    __warn_dataloader_iter_limitations(model)


def __verify_train_val_loop_configuration(trainer: "pl.Trainer", model: "pl.LightningModule") -> None:
    # verify minimum training requirements
    has_training_step = is_overridden("training_step", model)
    if not has_training_step:
        raise MisconfigurationException(
            "No `training_step()` method defined. Lightning `Trainer` expects as minimum a"
            " `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined."
        )
    has_optimizers = is_overridden("configure_optimizers", model)
    if not has_optimizers:
        raise MisconfigurationException(
            "No `configure_optimizers()` method defined. Lightning `Trainer` expects as minimum a"
            " `training_step()`, `train_dataloader()` and `configure_optimizers()` to be defined."
        )

    # verify minimum validation requirements
    has_val_loader = trainer.fit_loop.epoch_loop.val_loop._data_source.is_defined()
    has_val_step = is_overridden("validation_step", model)
    if has_val_loader and not has_val_step:
        rank_zero_warn("You passed in a `val_dataloader` but have no `validation_step`. Skipping val loop.")
    if has_val_step and not has_val_loader:
        rank_zero_warn(
            "You defined a `validation_step` but have no `val_dataloader`. Skipping val loop.",
            category=PossibleUserWarning,
        )

    # check legacy hooks are not present
    if callable(getattr(model, "training_epoch_end", None)):
        raise NotImplementedError(
            f"Support for `training_epoch_end` has been removed in v2.0.0. `{type(model).__name__}` implements this"
            " method. You can use the `on_train_epoch_end` hook instead. To access outputs, save them in-memory as"
            " instance attributes."
            " You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520."
        )
    if callable(getattr(model, "validation_epoch_end", None)):
        raise NotImplementedError(
            f"Support for `validation_epoch_end` has been removed in v2.0.0. `{type(model).__name__}` implements this"
            " method. You can use the `on_validation_epoch_end` hook instead. To access outputs, save them in-memory as"
            " instance attributes."
            " You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520."
        )


def __verify_eval_loop_configuration(model: "pl.LightningModule", stage: str) -> None:
    step_name = "validation_step" if stage == "val" else f"{stage}_step"
    has_step = is_overridden(step_name, model)

    # predict_step is not required to be overridden
    if stage == "predict":
        if model.predict_step is None:
            raise MisconfigurationException("`predict_step` cannot be None to run `Trainer.predict`")
        if not has_step and not is_overridden("forward", model):
            raise MisconfigurationException("`Trainer.predict` requires `forward` method to run.")
    else:
        # verify minimum evaluation requirements
        if not has_step:
            trainer_method = "validate" if stage == "val" else stage
            raise MisconfigurationException(f"No `{step_name}()` method defined to run `Trainer.{trainer_method}`.")

        # check legacy hooks are not present
        epoch_end_name = "validation_epoch_end" if stage == "val" else "test_epoch_end"
        if callable(getattr(model, epoch_end_name, None)):
            raise NotImplementedError(
                f"Support for `{epoch_end_name}` has been removed in v2.0.0. `{type(model).__name__}` implements this"
                f" method. You can use the `on_{epoch_end_name}` hook instead. To access outputs, save them in-memory"
                " as instance attributes."
                " You can find migration examples in https://github.com/Lightning-AI/lightning/pull/16520."
            )


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


def __warn_dataloader_iter_limitations(model: "pl.LightningModule") -> None:
    """Check if `dataloader_iter is enabled`."""
    if any(
        is_param_in_hook_signature(step_fn, "dataloader_iter", explicit=True)
        for step_fn in (model.training_step, model.validation_step, model.predict_step, model.test_step)
        if step_fn is not None
    ):
        rank_zero_warn(
            "You are using the `dataloader_iter` step flavor. If you consume the iterator more than once per step, the"
            " `batch_idx` argument in any hook that takes it will not match with the batch index of the last batch"
            " consumed. This might have unforeseen effects on callbacks or code that expects to get the correct index."
            " This will also not work well with gradient accumulation. This feature is very experimental and subject to"
            " change. Here be dragons.",
            category=PossibleUserWarning,
        )


def __verify_configure_model_configuration(model: "pl.LightningModule") -> None:
    if is_overridden("configure_sharded_model", model):
        name = type(model).__name__
        if is_overridden("configure_model", model):
            raise RuntimeError(
                f"Both `{name}.configure_model`, and `{name}.configure_sharded_model` are overridden. The latter is"
                f" deprecated and it should be replaced with the former."
            )
        rank_zero_deprecation(
            f"You have overridden `{name}.configure_sharded_model` which is deprecated. Please override the"
            " `configure_model` hook instead. Instantiation with the newer hook will be created on the device right"
            " away and have the right data type depending on the precision setting in the Trainer."
        )
