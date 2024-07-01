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
"""Utilities for loggers."""

from pathlib import Path
from typing import Any, List, Tuple, Union

from torch import Tensor

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Checkpoint, ModelCheckpoint


def _version(loggers: List[Any], separator: str = "_") -> Union[int, str]:
    if len(loggers) == 1:
        return loggers[0].version
    # Concatenate versions together, removing duplicates and preserving order
    return separator.join(dict.fromkeys(str(logger.version) for logger in loggers))


def _scan_checkpoints(
    checkpoint_callback: Checkpoint, logged_model_time: dict, include_distributed_checkpoints: bool = False
) -> List[Tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    Args:
        checkpoint_callback: Checkpoint callback reference.
        logged_model_time: dictionary containing the logged model times.
        include_distributed_checkpoints: flag to include distributed directories.

    Returns:
        List of tuples containing the time, path, score, and tag of the checkpoints.

    """
    # get checkpoints to be saved with associated score
    checkpoints = {}
    if hasattr(checkpoint_callback, "last_model_path") and hasattr(checkpoint_callback, "current_score"):
        checkpoints[checkpoint_callback.last_model_path] = (checkpoint_callback.current_score, "latest")

    if hasattr(checkpoint_callback, "best_model_path") and hasattr(checkpoint_callback, "best_model_score"):
        checkpoints[checkpoint_callback.best_model_path] = (checkpoint_callback.best_model_score, "best")

    if hasattr(checkpoint_callback, "best_k_models"):
        for key, value in checkpoint_callback.best_k_models.items():
            checkpoints[key] = (value, "best_k")

    checkpoints = sorted(
        (Path(p).stat().st_mtime, p, s, tag)
        for p, (s, tag) in checkpoints.items()
        if Path(p).is_file() or (include_distributed_checkpoints and Path(p).is_dir())
    )
    checkpoints = [c for c in checkpoints if c[1] not in logged_model_time or logged_model_time[c[1]] < c[0]]
    return checkpoints


def _log_hyperparams(trainer: "pl.Trainer") -> None:
    if not trainer.loggers:
        return

    pl_module = trainer.lightning_module
    datamodule_log_hyperparams = trainer.datamodule._log_hyperparams if trainer.datamodule is not None else False

    hparams_initial = None
    if pl_module._log_hyperparams and datamodule_log_hyperparams:
        datamodule_hparams = trainer.datamodule.hparams_initial
        lightning_hparams = pl_module.hparams_initial
        inconsistent_keys = []
        for key in lightning_hparams.keys() & datamodule_hparams.keys():
            lm_val, dm_val = lightning_hparams[key], datamodule_hparams[key]
            if (
                type(lm_val) != type(dm_val)
                or (isinstance(lm_val, Tensor) and id(lm_val) != id(dm_val))
                or lm_val != dm_val
            ):
                inconsistent_keys.append(key)
        if inconsistent_keys:
            raise RuntimeError(
                f"Error while merging hparams: the keys {inconsistent_keys} are present "
                "in both the LightningModule's and LightningDataModule's hparams "
                "but have different values."
            )
        hparams_initial = {**lightning_hparams, **datamodule_hparams}
    elif pl_module._log_hyperparams:
        hparams_initial = pl_module.hparams_initial
    elif datamodule_log_hyperparams:
        hparams_initial = trainer.datamodule.hparams_initial

    for logger in trainer.loggers:
        if hparams_initial is not None:
            logger.log_hyperparams(hparams_initial)
        logger.log_graph(pl_module)
        logger.save()


def _generate_checkpoint_identifier(checkpoint_callback: ModelCheckpoint) -> str:
    parts = []

    # Include monitor, mode, and save_top_k if they are not None
    if hasattr(checkpoint_callback, "monitor") and checkpoint_callback.monitor is not None:
        parts.append(f"monitor={checkpoint_callback.monitor}")
    if hasattr(checkpoint_callback, "mode") and checkpoint_callback.mode is not None:
        parts.append(f"mode={checkpoint_callback.mode}")
    if hasattr(checkpoint_callback, "save_top_k"):
        parts.append(f"save_top_k={checkpoint_callback.save_top_k}")

    # Frequency of saving based on training steps or epochs
    if hasattr(checkpoint_callback, "_every_n_train_steps") and checkpoint_callback._every_n_train_steps:
        parts.append(f"every_n_train_steps={checkpoint_callback._every_n_train_steps}")
    if hasattr(checkpoint_callback, "every_n_epochs") and checkpoint_callback.every_n_epochs:
        parts.append(f"every_n_epochs={checkpoint_callback.every_n_epochs}")

    # Time interval for saving, if applicable
    # if hasattr(checkpoint_callback, 'train_time_interval') and checkpoint_callback.train_time_interval:
    #     # Assuming train_time_interval is a timedelta object or similar
    #     time_str = f"{checkpoint_callback.train_time_interval.total_seconds()}s"
    #     parts.append(f"time_interval={time_str}")

    # Custom file naming, if used
    # if hasattr(checkpoint_callback, 'filename') and checkpoint_callback.filename:
    #     parts.append(f"filename={checkpoint_callback.filename}")

    # Ensure there's at least one part in the identifier; use a default if not
    if not parts:
        parts.append("default_checkpoint")

    return "__".join(parts)
