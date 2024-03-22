# Copyright Lightning AI.
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
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, Optional, Type, Union

from packaging.version import Version

import lightning.pytorch as pl
from lightning.fabric.utilities.device_dtype_mixin import _DeviceDtypeModuleMixin
from lightning.pytorch.callbacks import Checkpoint, EarlyStopping
from lightning.pytorch.trainer.states import TrainerStatus
from lightning.pytorch.utilities.exceptions import _TunerExitException
from lightning.pytorch.utilities.model_helpers import is_overridden
from lightning.pytorch.utilities.rank_zero import rank_zero_warn

log = logging.getLogger(__name__)


def _call_and_handle_interrupt(trainer: "pl.Trainer", trainer_fn: Callable, *args: Any, **kwargs: Any) -> Any:
    r"""Error handling, intended to be used only for main trainer function entry points (fit, validate, test, predict)
    as all errors should funnel through them.

    Args:
        trainer_fn: one of (fit, validate, test, predict)
        *args: positional arguments to be passed to the `trainer_fn`
        **kwargs: keyword arguments to be passed to `trainer_fn`

    """
    try:
        if trainer.strategy.launcher is not None:
            return trainer.strategy.launcher.launch(trainer_fn, *args, trainer=trainer, **kwargs)
        return trainer_fn(*args, **kwargs)

    except _TunerExitException:
        _call_teardown_hook(trainer)
        trainer._teardown()
        trainer.state.status = TrainerStatus.FINISHED
        trainer.state.stage = None

    # TODO: Unify both exceptions below, where `KeyboardError` doesn't re-raise
    except KeyboardInterrupt as exception:
        rank_zero_warn("Detected KeyboardInterrupt, attempting graceful shutdown...")
        # user could press Ctrl+c many times... only shutdown once
        if not trainer.interrupted:
            _interrupt(trainer, exception)
    except BaseException as exception:
        _interrupt(trainer, exception)
        trainer._teardown()
        # teardown might access the stage so we reset it after
        trainer.state.stage = None
        raise


def _interrupt(trainer: "pl.Trainer", exception: BaseException) -> None:
    trainer.state.status = TrainerStatus.INTERRUPTED
    _call_callback_hooks(trainer, "on_exception", exception)
    if trainer.datamodule is not None:
        _call_lightning_datamodule_hook(trainer, "on_exception", exception)
    trainer.strategy.on_exception(exception)
    for logger in trainer.loggers:
        logger.finalize("failed")


def _call_setup_hook(trainer: "pl.Trainer") -> None:
    assert trainer.state.fn is not None
    fn = trainer.state.fn

    # It is too early to move the model to the device, but we fake the `LightningModule.device` property
    # so the user can access it in the `LightningModule.setup` hook
    for module in trainer.lightning_module.modules():
        if isinstance(module, _DeviceDtypeModuleMixin):
            module._device = trainer.strategy.root_device

    # Trigger lazy creation of experiment in loggers so loggers have their metadata available
    for logger in trainer.loggers:
        if hasattr(logger, "experiment"):
            _ = logger.experiment

    trainer.strategy.barrier("pre_setup")

    if trainer.datamodule is not None:
        _call_lightning_datamodule_hook(trainer, "setup", stage=fn)
    _call_callback_hooks(trainer, "setup", stage=fn)
    _call_lightning_module_hook(trainer, "setup", stage=fn)

    trainer.strategy.barrier("post_setup")


def _call_configure_model(trainer: "pl.Trainer") -> None:
    # legacy hook
    if is_overridden("configure_sharded_model", trainer.lightning_module):
        with trainer.strategy.model_sharded_context():
            _call_lightning_module_hook(trainer, "configure_sharded_model")

    # we don't normally check for this before calling the hook. it is done here to avoid instantiating the context
    # managers
    if is_overridden("configure_model", trainer.lightning_module):
        with trainer.strategy.tensor_init_context(), trainer.strategy.model_sharded_context(), trainer.precision_plugin.module_init_context():  # noqa: E501
            _call_lightning_module_hook(trainer, "configure_model")


def _call_teardown_hook(trainer: "pl.Trainer") -> None:
    assert trainer.state.fn is not None
    fn = trainer.state.fn

    if trainer.datamodule is not None:
        _call_lightning_datamodule_hook(trainer, "teardown", stage=fn)

    _call_callback_hooks(trainer, "teardown", stage=fn)
    _call_lightning_module_hook(trainer, "teardown", stage=fn)

    trainer.lightning_module._current_fx_name = None
    # these could have become stale if metrics are defined in `setup`
    trainer.lightning_module._metric_attributes = None

    # todo: TPU 8 cores hangs in flush with TensorBoard. Might do for all loggers.
    # It might be related to xla tensors blocked when moving the cpu kill loggers.
    for logger in trainer.loggers:
        logger.finalize("success")

    # summarize profile results
    trainer.profiler.describe()


def _call_lightning_module_hook(
    trainer: "pl.Trainer",
    hook_name: str,
    *args: Any,
    pl_module: Optional["pl.LightningModule"] = None,
    **kwargs: Any,
) -> Any:
    log.debug(f"{trainer.__class__.__name__}: calling lightning module hook: {hook_name}")

    pl_module = pl_module or trainer.lightning_module

    if pl_module is None:
        raise TypeError("No `LightningModule` is available to call hooks on.")

    fn = getattr(pl_module, hook_name)
    if not callable(fn):
        return None

    prev_fx_name = pl_module._current_fx_name
    pl_module._current_fx_name = hook_name

    with trainer.profiler.profile(f"[LightningModule]{pl_module.__class__.__name__}.{hook_name}"):
        output = fn(*args, **kwargs)

    # restore current_fx when nested context
    pl_module._current_fx_name = prev_fx_name

    return output


def _call_lightning_datamodule_hook(
    trainer: "pl.Trainer",
    hook_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    log.debug(f"{trainer.__class__.__name__}: calling lightning datamodule hook: {hook_name}")

    if trainer.datamodule is None:
        raise TypeError("No `LightningDataModule` is available to call hooks on.")

    fn = getattr(trainer.datamodule, hook_name)
    if callable(fn):
        with trainer.profiler.profile(f"[LightningDataModule]{trainer.datamodule.__class__.__name__}.{hook_name}"):
            return fn(*args, **kwargs)
    return None


def _call_callback_hooks(
    trainer: "pl.Trainer",
    hook_name: str,
    *args: Any,
    monitoring_callbacks: Optional[bool] = None,
    **kwargs: Any,
) -> None:
    log.debug(f"{trainer.__class__.__name__}: calling callback hook: {hook_name}")

    pl_module = trainer.lightning_module
    if pl_module:
        prev_fx_name = pl_module._current_fx_name
        pl_module._current_fx_name = hook_name

    callbacks = trainer.callbacks
    if monitoring_callbacks is True:
        # the list of "monitoring callbacks" is hard-coded to these two. we could add an API to define this
        callbacks = [cb for cb in callbacks if isinstance(cb, (EarlyStopping, Checkpoint))]
    elif monitoring_callbacks is False:
        callbacks = [cb for cb in callbacks if not isinstance(cb, (EarlyStopping, Checkpoint))]

    for callback in callbacks:
        fn = getattr(callback, hook_name)
        if callable(fn):
            with trainer.profiler.profile(f"[Callback]{callback.state_key}.{hook_name}"):
                fn(trainer, trainer.lightning_module, *args, **kwargs)

    if pl_module:
        # restore current_fx when nested context
        pl_module._current_fx_name = prev_fx_name


def _call_callbacks_state_dict(trainer: "pl.Trainer") -> Dict[str, dict]:
    """Called when saving a model checkpoint, calls and returns every callback's `state_dict`, keyed by
    `Callback.state_key`."""
    callback_state_dicts = {}
    for callback in trainer.callbacks:
        state_dict = callback.state_dict()
        if state_dict:
            callback_state_dicts[callback.state_key] = state_dict
    return callback_state_dicts


def _call_callbacks_on_save_checkpoint(trainer: "pl.Trainer", checkpoint: Dict[str, Any]) -> None:
    """Called when saving a model checkpoint, calls every callback's `on_save_checkpoint` hook."""
    pl_module = trainer.lightning_module
    if pl_module:
        prev_fx_name = pl_module._current_fx_name
        pl_module._current_fx_name = "on_save_checkpoint"

    for callback in trainer.callbacks:
        with trainer.profiler.profile(f"[Callback]{callback.state_key}.on_save_checkpoint"):
            callback.on_save_checkpoint(trainer, trainer.lightning_module, checkpoint)

    if pl_module:
        # restore current_fx when nested context
        pl_module._current_fx_name = prev_fx_name


def _call_callbacks_on_load_checkpoint(trainer: "pl.Trainer", checkpoint: Dict[str, Any]) -> None:
    """Called when loading a model checkpoint.

    Calls every callback's `on_load_checkpoint` hook. We have a dedicated function for this rather than using
    `_call_callback_hooks` because we have special logic for getting callback_states.

    """
    pl_module = trainer.lightning_module
    if pl_module:
        prev_fx_name = pl_module._current_fx_name
        pl_module._current_fx_name = "on_load_checkpoint"

    callback_states: Optional[Dict[Union[Type, str], Dict]] = checkpoint.get("callbacks")

    if callback_states is None:
        return

    is_legacy_ckpt = Version(checkpoint["pytorch-lightning_version"]) < Version("1.5.0dev")
    current_callbacks_keys = {cb._legacy_state_key if is_legacy_ckpt else cb.state_key for cb in trainer.callbacks}
    difference = callback_states.keys() - current_callbacks_keys
    if difference:
        rank_zero_warn(
            "Be aware that when using `ckpt_path`,"
            " callbacks used to create the checkpoint need to be provided during `Trainer` instantiation."
            f" Please add the following callbacks: {list(difference)}.",
        )

    for callback in trainer.callbacks:
        with trainer.profiler.profile(f"[Callback]{callback.state_key}.on_load_checkpoint"):
            callback.on_load_checkpoint(trainer, trainer.lightning_module, checkpoint)

    if pl_module:
        # restore current_fx when nested context
        pl_module._current_fx_name = prev_fx_name


def _call_callbacks_load_state_dict(trainer: "pl.Trainer", checkpoint: Dict[str, Any]) -> None:
    """Called when loading a model checkpoint, calls every callback's `load_state_dict`."""
    callback_states: Optional[Dict[Union[Type, str], Dict]] = checkpoint.get("callbacks")

    if callback_states is None:
        return

    for callback in trainer.callbacks:
        state = callback_states.get(callback.state_key, callback_states.get(callback._legacy_state_key))
        if state:
            state = deepcopy(state)
            callback.load_state_dict(state)


def _call_strategy_hook(
    trainer: "pl.Trainer",
    hook_name: str,
    *args: Any,
    **kwargs: Any,
) -> Any:
    log.debug(f"{trainer.__class__.__name__}: calling strategy hook: {hook_name}")

    pl_module = trainer.lightning_module
    prev_fx_name = pl_module._current_fx_name
    pl_module._current_fx_name = hook_name

    fn = getattr(trainer.strategy, hook_name)
    if not callable(fn):
        return None

    with trainer.profiler.profile(f"[Strategy]{trainer.strategy.__class__.__name__}.{hook_name}"):
        output = fn(*args, **kwargs)

    # restore current_fx when nested context
    pl_module._current_fx_name = prev_fx_name

    return output
