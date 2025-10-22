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

from collections.abc import ItemsView, KeysView, Mapping, ValuesView
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

from torch import Tensor
from typing_extensions import Self

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Checkpoint


def _version(loggers: list[Any], separator: str = "_") -> Union[int, str]:
    if len(loggers) == 1:
        return loggers[0].version
    # Concatenate versions together, removing duplicates and preserving order
    return separator.join(dict.fromkeys(str(logger.version) for logger in loggers))


def _scan_checkpoints(checkpoint_callback: Checkpoint, logged_model_time: dict) -> list[tuple[float, str, float, str]]:
    """Return the checkpoints to be logged.

    Args:
        checkpoint_callback: Checkpoint callback reference.
        logged_model_time: dictionary containing the logged model times.

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
        (Path(p).stat().st_mtime, p, s, tag) for p, (s, tag) in checkpoints.items() if Path(p).is_file()
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
            if key == "_class_path":
                # Skip LightningCLI's internal hparam
                continue
            lm_val, dm_val = lightning_hparams[key], datamodule_hparams[key]
            if (
                type(lm_val) != type(dm_val)  # noqa: E721
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

    # Don't log LightningCLI's internal hparam
    if hparams_initial is not None:
        hparams_initial = {k: v for k, v in hparams_initial.items() if k != "_class_path"}

    for logger in trainer.loggers:
        if hparams_initial is not None:
            logger.log_hyperparams(hparams_initial)
        logger.log_graph(pl_module)
        logger.save()


_T = TypeVar("_T")


class _ListMap(list[_T]):
    """A hybrid container for loggers allowing both index and name access."""

    def __init__(self, loggers: Union[list[_T], Mapping[str, _T]] = None):
        if isinstance(loggers, Mapping):
            # super inits list with values
            if any(not isinstance(x, str) for x in loggers):
                raise TypeError("When providing a Mapping, all keys must be of type str.")
            super().__init__(loggers.values())
            self._dict = dict(zip(loggers.keys(), range(len(loggers))))
        else:
            super().__init__(() if loggers is None else loggers)
            self._dict: dict = {}

    def __eq__(self, other: Any) -> bool:
        self_list = list(self)
        if isinstance(other, _ListMap):
            return self_list == list(other) and self._dict == other._dict
        if isinstance(other, list):
            return self_list == other
        return False

    # --- List-like interface ---
    def __getitem__(self, key: Union[int, slice, str]) -> _T:
        if isinstance(key, (int, slice)):
            return list.__getitem__(self, key)
        if isinstance(key, str):
            return list.__getitem__(self, self._dict[key])
        raise TypeError("Key must be int / slice (for index) or str (for name).")

    def __add__(self, other: Union[list[_T], Self]) -> list[_T]:
        # todo
        return list.__add__(self, other)

    def __iadd__(self, other: Union[list[_T], Self]) -> Self:
        # todo
        return list.__iadd__(self, other)

    def __setitem__(self, key: Union[int, slice, str], value: _T) -> None:
        if isinstance(key, (int, slice)):
            # replace element by index
            return list.__setitem__(self, key, value)
        if isinstance(key, str):
            # replace or insert by name
            if key in self._dict:
                list.__setitem__(self, self._dict[key], value)
            else:
                self.append(value)
                self._dict[key] = len(self) - 1
            return None
        raise TypeError("Key must be int or str")

    def __contains__(self, item: Union[_T, str]) -> bool:
        if isinstance(item, str):
            return item in self._dict
        return list.__contains__(self, item)

    # --- Dict-like interface ---

    def __delitem__(self, key: Union[int, slice, str]) -> None:
        if isinstance(key, (int, slice)):
            loggers = list.__getitem__(self, key)
            super(list, self).__delitem__(key)
            for logger in loggers if isinstance(key, slice) else [loggers]:
                name = getattr(logger, "name", None)
                if name:
                    self._dict.pop(name, None)
        elif isinstance(key, str):
            logger = self._dict.pop(key)
            self.remove(logger)
        else:
            raise TypeError("Key must be int or str")

    def keys(self) -> KeysView[str]:
        return self._dict.keys()

    def values(self) -> ValuesView[_T]:
        d = {k: self[v] for k, v in self._dict.items()}
        return d.values()

    def items(self) -> ItemsView[str, _T]:
        d = {k: self[v] for k, v in self._dict.items()}
        return d.items()

    # --- List and Dict interface ---
    def pop(self, key: Union[int, str] = -1, default: Optional[Any] = None) -> _T:
        if isinstance(key, int):
            ret = list.pop(self, key)
            for str_key, idx in list(self._dict.items()):
                if idx == key:
                    self._dict.pop(str_key)
                elif idx > key:
                    self._dict[str_key] = idx - 1
            return ret
        if isinstance(key, str):
            if key not in self._dict:
                return default
            return self.pop(self._dict[key])
        raise TypeError("Key must be int or str")

    def __repr__(self) -> str:
        ret = super().__repr__()
        return f"_ListMap({ret}, keys={list(self._dict.keys())})"
