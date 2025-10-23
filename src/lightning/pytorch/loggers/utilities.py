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

from collections.abc import ItemsView, Iterable, KeysView, Mapping, ValuesView
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, SupportsIndex, TypeVar, Union

from torch import Tensor
from typing_extensions import Self

import lightning.pytorch as pl
from lightning.pytorch.callbacks import Checkpoint

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison


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
    """A hybrid container allowing both index and name access.

    This class extends the built-in list to provide dictionary-like access to its elements
    using string keys. It maintains an internal mapping of string keys to list indices,
    allowing users to retrieve, set, and delete elements by their associated names.

    Args:
        __iterable (Union[Iterable[_T], Mapping[str, _T]], optional): An iterable of objects or a mapping
            of string keys to __iterable to initialize the container.

    Raises:
        TypeError: If a Mapping is provided and any of its keys are not of type str.

    Example:
        >>> listmap = _ListMap({'obj1': 1, 'obj2': 2})
        >>> listmap['obj1']  # Access by name
        1
        >>> listmap[0]  # Access by index
        1
        >>> listmap['obj2'] = 3  # Set by name
        >>> listmap[1]  # Now returns obj3
        3
        >>> listmap.append(4)  # Append by index
        >>> listmap[2]
        4

    """

    def __init__(self, __iterable: Union[Mapping[str, _T], Iterable[_T]] = None):
        if isinstance(__iterable, Mapping):
            # super inits list with values
            if any(not isinstance(x, str) for x in __iterable):
                raise TypeError("When providing a Mapping, all keys must be of type str.")
            super().__init__(__iterable.values())
            self._dict = dict(zip(__iterable.keys(), range(len(__iterable))))
        else:
            default_dict = {}
            if isinstance(__iterable, _ListMap):
                default_dict = __iterable._dict.copy()
            super().__init__(() if __iterable is None else __iterable)
            self._dict: dict = default_dict

    def __eq__(self, other: Any) -> bool:
        list_eq = list.__eq__(self, other)
        if isinstance(other, _ListMap):
            return list_eq and self._dict == other._dict
        return list_eq

    def copy(self):
        new_listmap = _ListMap(self)
        new_listmap._dict = self._dict.copy()
        return new_listmap

    def extend(self, __iterable: Iterable[_T]) -> None:
        if isinstance(__iterable, _ListMap):
            offset = len(self)
            for key, idx in __iterable._dict.items():
                self._dict[key] = idx + offset
        super().extend(__iterable)

    def pop(self, key: Union[SupportsIndex, str] = -1, default: Optional[Any] = None) -> _T:
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

    def insert(self, index: SupportsIndex, __object: _T) -> None:
        for key, idx in self._dict.items():
            if idx >= index:
                self._dict[key] = idx + 1
        list.insert(self, index, __object)

    def remove(self, __object: _T) -> None:
        idx = self.index(__object)
        name = None
        for key, val in self._dict.items():
            if val == idx:
                name = key
            elif val > idx:
                self._dict[key] = val - 1
        if name:
            self._dict.pop(name, None)
        list.remove(self, __object)

    def sort(
        self,
        *,
        key: Optional[Callable[[_T], "SupportsRichComparison"]] = None,
        reverse: bool = False,
    ) -> None:
        # Create a mapping from item to its name(s)
        item_to_names = {}
        for name, idx in self._dict.items():
            item = self[idx]
            item_to_names.setdefault(item, []).append(name)
        # Sort the list
        list.sort(self, key=key, reverse=reverse)
        # Update _dict with new indices
        new_dict = {}
        for idx, item in enumerate(self):
            if item in item_to_names:
                for name in item_to_names[item]:
                    new_dict[name] = idx
        self._dict = new_dict

    # --- List-like interface ---
    def __getitem__(self, key: Union[int, slice, str]) -> _T:
        if isinstance(key, str):
            return self[self._dict[key]]
        return list.__getitem__(self, key)

    def __add__(self, other: Union[list[_T], Self]) -> Self:
        new_listmap = self.copy()
        new_listmap += other
        return new_listmap

    def __iadd__(self, other: Union[list[_T], Self]) -> Self:
        if isinstance(other, _ListMap):
            offset = len(self)
            for key, idx in other._dict.items():
                # notes: if there are duplicate keys, the ones from other will overwrite self
                self._dict[key] = idx + offset

        return super().__iadd__(other)

    def __setitem__(self, key: Union[SupportsIndex, slice, str], value: _T) -> None:
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
            list.__delitem__(self, key)
            for _key in key.indices(len(self)) if isinstance(key, slice) else [key]:
                # update indices in the dict
                for str_key, idx in list(self._dict.items()):
                    if idx == _key:
                        self._dict.pop(str_key)
                    elif idx > _key:
                        self._dict[str_key] = idx - 1
        elif isinstance(key, str):
            if key not in self._dict:
                raise KeyError(f"Key '{key}' not found.")
            self.__delitem__(self._dict[key])
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

    def get(self, __key: str, default: Optional[Any] = None) -> _T:
        if __key in self._dict:
            return self[self._dict[__key]]
        return default

    def __repr__(self) -> str:
        ret = super().__repr__()
        return f"_ListMap({ret}, keys={list(self._dict.keys())})"

    def reverse(self) -> None:
        for key, idx in self._dict.items():
            self._dict[key] = len(self) - 1 - idx
        list.reverse(self)

    def clear(self):
        self._dict.clear()
        list.clear(self)
