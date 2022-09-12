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
# limitations under the License
import logging
import os
import uuid
from typing import Any, Dict, Optional, Tuple

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import DummyLogger
from pytorch_lightning.utilities.data import has_len_all_ranks
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.memory import garbage_collection_cuda, is_oom_error
from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_hasattr, lightning_setattr
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

log = logging.getLogger(__name__)


def scale_batch_size(
    trainer: "pl.Trainer",
    model: "pl.LightningModule",
    mode: str = "power",
    steps_per_trial: int = 3,
    init_val: int = 2,
    max_trials: int = 25,
    batch_arg_name: str = "batch_size",
) -> Optional[int]:
    """See :meth:`~pytorch_lightning.tuner.tuning.Tuner.scale_batch_size`"""
    if trainer.fast_dev_run:
        rank_zero_warn("Skipping batch size scaler since fast_dev_run is enabled.")
        return

    if not lightning_hasattr(model, batch_arg_name):
        raise MisconfigurationException(f"Field {batch_arg_name} not found in both `model` and `model.hparams`")
    if hasattr(model, batch_arg_name) and hasattr(model, "hparams") and batch_arg_name in model.hparams:
        rank_zero_warn(
            f"Field `model.{batch_arg_name}` and `model.hparams.{batch_arg_name}` are mutually exclusive!"
            f" `model.{batch_arg_name}` will be used as the initial batch size for scaling."
            " If this is not the intended behavior, please remove either one."
        )

    if not trainer._data_connector._train_dataloader_source.is_module():
        raise MisconfigurationException(
            "The batch scaling feature cannot be used with dataloaders passed directly to `.fit()`."
            " Please disable the feature or incorporate the dataloader into the model."
        )

    # Save initial model, that is loaded after batch size is found
    ckpt_path = os.path.join(trainer.default_root_dir, f".scale_batch_size_{uuid.uuid4()}.ckpt")
    trainer.save_checkpoint(ckpt_path)
    params = __scale_batch_dump_params(trainer)

    # Set to values that are required by the algorithm
    __scale_batch_reset_params(trainer, steps_per_trial)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.disable()

    # Initially we just double in size until an OOM is encountered
    new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=init_val)  # initially set to init_val
    if mode == "power":
        new_size = _run_power_scaling(trainer, model, new_size, batch_arg_name, max_trials)
    elif mode == "binsearch":
        new_size = _run_binsearch_scaling(trainer, model, new_size, batch_arg_name, max_trials)
    else:
        raise ValueError("mode in method `scale_batch_size` could either be `power` or `binsearch`")

    garbage_collection_cuda()
    log.info(f"Finished batch size finder, will continue with full run using batch size {new_size}")

    __scale_batch_restore_params(trainer, params)

    if trainer.progress_bar_callback:
        trainer.progress_bar_callback.enable()

    # Restore initial state of model
    trainer._checkpoint_connector.restore(ckpt_path)
    trainer.strategy.remove_checkpoint(ckpt_path)

    return new_size


def __scale_batch_dump_params(trainer: "pl.Trainer") -> Dict[str, Any]:
    return {
        "max_steps": trainer.fit_loop.max_steps,
        "logger": trainer.logger,
        "callbacks": trainer.callbacks,
        "auto_scale_batch_size": trainer.auto_scale_batch_size,
        "auto_lr_find": trainer.auto_lr_find,
        "limit_train_batches": trainer.limit_train_batches,
    }


def __scale_batch_reset_params(trainer: "pl.Trainer", steps_per_trial: int) -> None:
    trainer.auto_scale_batch_size = None  # prevent recursion
    trainer.auto_lr_find = False  # avoid lr find being called multiple times
    trainer.fit_loop.max_steps = steps_per_trial  # take few steps
    trainer.loggers = [DummyLogger()] if trainer.loggers else []
    trainer.callbacks = []  # not needed before full run
    trainer.limit_train_batches = 1.0


def __scale_batch_restore_params(trainer: "pl.Trainer", params: Dict[str, Any]) -> None:
    trainer.auto_scale_batch_size = params["auto_scale_batch_size"]
    trainer.auto_lr_find = params["auto_lr_find"]
    trainer.fit_loop.max_steps = params["max_steps"]
    trainer.logger = params["logger"]
    trainer.callbacks = params["callbacks"]
    trainer.limit_train_batches = params["limit_train_batches"]


def _run_power_scaling(
    trainer: "pl.Trainer", model: "pl.LightningModule", new_size: int, batch_arg_name: str, max_trials: int
) -> int:
    """Batch scaling mode where the size is doubled at each iteration until an OOM error is encountered."""
    # this flag is used to determine whether the previously scaled batch size, right before OOM, was a success or not
    # if it was we exit, else we continue downscaling in case we haven't encountered a single optimal batch size
    any_success = False
    for _ in range(max_trials):
        garbage_collection_cuda()

        # reset after each try
        _reset_progress(trainer)

        try:
            # Try fit
            trainer.tuner._run(model)
            # Double in size
            new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc="succeeded")

            if not changed:
                break

            # Force the train dataloader to reset as the batch size has changed
            trainer.reset_train_dataloader(model)
            trainer.reset_val_dataloader(model)
            any_success = True
        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, factor=0.5, desc="failed")
                # Force the train dataloader to reset as the batch size has changed
                trainer.reset_train_dataloader(model)
                trainer.reset_val_dataloader(model)
                if any_success:
                    break
            else:
                raise  # some other error not memory related

    return new_size


def _run_binsearch_scaling(
    trainer: "pl.Trainer", model: "pl.LightningModule", new_size: int, batch_arg_name: str, max_trials: int
) -> int:
    """Batch scaling mode where the size is initially is doubled at each iteration until an OOM error is
    encountered.

    Hereafter, the batch size is further refined using a binary search
    """
    low = 1
    high = None
    count = 0
    while True:
        garbage_collection_cuda()

        # reset after each try
        _reset_progress(trainer)

        try:
            # Try fit
            trainer.tuner._run(model)
            count += 1
            if count > max_trials:
                break
            # Double in size
            low = new_size
            if high:
                if high - low <= 1:
                    break
                midval = (high + low) // 2
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc="succeeded")
            else:
                new_size, changed = _adjust_batch_size(trainer, batch_arg_name, factor=2.0, desc="succeeded")

            if not changed:
                break

            # Force the train dataloader to reset as the batch size has changed
            trainer.reset_train_dataloader(model)
            trainer.reset_val_dataloader(model)

        except RuntimeError as exception:
            # Only these errors should trigger an adjustment
            if is_oom_error(exception):
                # If we fail in power mode, half the size and return
                garbage_collection_cuda()
                high = new_size
                midval = (high + low) // 2
                new_size, _ = _adjust_batch_size(trainer, batch_arg_name, value=midval, desc="failed")

                # Force the train dataloader to reset as the batch size has changed
                trainer.reset_train_dataloader(model)
                trainer.reset_val_dataloader(model)

                if high - low <= 1:
                    break
            else:
                raise  # some other error not memory related

    return new_size


def _adjust_batch_size(
    trainer: "pl.Trainer",
    batch_arg_name: str = "batch_size",
    factor: float = 1.0,
    value: Optional[int] = None,
    desc: Optional[str] = None,
) -> Tuple[int, bool]:
    """Helper function for adjusting the batch size.

    Args:
        trainer: instance of pytorch_lightning.Trainer

        batch_arg_name: name of the field where batch_size is stored.

        factor: value which the old batch size is multiplied by to get the
            new batch size

        value: if a value is given, will override the batch size with this value.
            Note that the value of `factor` will not have an effect in this case

        desc: either `succeeded` or `failed`. Used purely for logging

    Returns:
        The new batch size for the next trial and a bool that signals whether the
        new value is different than the previous batch size.
    """
    model = trainer.lightning_module
    batch_size = lightning_getattr(model, batch_arg_name)
    new_size = value if value is not None else int(batch_size * factor)
    if desc:
        log.info(f"Batch size {batch_size} {desc}, trying batch size {new_size}")

    if not _is_valid_batch_size(new_size, trainer.train_dataloader, trainer):
        new_size = min(new_size, len(trainer.train_dataloader.dataset))

    changed = new_size != batch_size
    lightning_setattr(model, batch_arg_name, new_size)
    return new_size, changed


def _is_valid_batch_size(batch_size: int, dataloader: DataLoader, trainer: "pl.Trainer"):
    module = trainer.lightning_module or trainer.datamodule
    return not has_len_all_ranks(dataloader, trainer.strategy, module) or batch_size <= len(dataloader)


def _reset_progress(trainer: "pl.Trainer") -> None:
    if trainer.lightning_module.automatic_optimization:
        trainer.fit_loop.epoch_loop.batch_loop.optimizer_loop.optim_progress.reset()
    else:
        trainer.fit_loop.epoch_loop.batch_loop.manual_loop.optim_step_progress.reset()

    trainer.fit_loop.epoch_progress.reset()
