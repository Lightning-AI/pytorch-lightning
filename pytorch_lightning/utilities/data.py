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

from typing import Any, Generator, Iterable, Mapping, Union

import torch
from torch.utils.data import DataLoader, IterableDataset

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.warnings import WarningCache

BType = Union[torch.Tensor, str, Mapping[Any, "BType"], Iterable["BType"]]

warning_cache = WarningCache()


def _extract_batch_size(batch: BType) -> Generator[int, None, None]:
    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            yield 1
        else:
            yield batch.size(0)
    elif isinstance(batch, str):
        yield len(batch)
    elif isinstance(batch, (Iterable, Mapping)):
        if isinstance(batch, Mapping):
            batch = batch.values()

        for sample in batch:
            yield from _extract_batch_size(sample)
    else:
        yield 1


def extract_batch_size(batch: BType) -> int:
    """Unpack a batch to find a ``torch.Tensor``.

    Returns:
        ``len(tensor)`` when found, or ``1`` when it hits an empty or non iterable.
    """
    batch_size = None
    for bs in _extract_batch_size(batch):
        if batch_size is None:
            batch_size = bs
        elif batch_size != bs:
            warning_cache.warn(
                "Trying to infer the `batch_size` from an ambiguous collection. The batch size we"
                f" found is {batch_size}. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`."
            )
            break

    return batch_size


def has_iterable_dataset(dataloader: DataLoader) -> bool:
    return hasattr(dataloader, "dataset") and isinstance(dataloader.dataset, IterableDataset)


def has_len(dataloader: Union[DataLoader, Iterable]) -> bool:
    """Checks if a given Dataloader has ``__len__`` method implemented i.e. if it is a finite dataloader or
    infinite dataloader.

    Raises:
        ValueError:
            If the length of Dataloader is 0, as it requires at least one batch
    """

    try:
        # try getting the length
        if len(dataloader) == 0:
            raise ValueError("`Dataloader` returned 0 length. Please make sure that it returns at least 1 batch")
        has_len = True
    except TypeError:
        has_len = False
    except NotImplementedError:  # e.g. raised by torchtext if a batch_size_fn is used
        has_len = False

    if has_len and has_iterable_dataset(dataloader):
        rank_zero_warn(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-process data loading (when num_workers > 1),"
            " `__len__` could be inaccurate if each worker is not configured independently"
            " to avoid having duplicate data."
        )
    return has_len


def has_len_all_ranks(
    dataloader: DataLoader,
    training_type: "pl.TrainingTypePlugin",
    model: Union["pl.LightningModule", "pl.LightningDataModule"],
) -> bool:
    """Checks if a given Dataloader has ``__len__`` method implemented i.e. if it is a finite dataloader or
    infinite dataloader.

    Raises:
        ValueError:
            If the length of Dataloader is 0, as it requires at least one batch
    """
    try:
        total_length = training_type.reduce(torch.tensor(len(dataloader)).to(model.device), reduce_op="sum")
        local_length = len(dataloader)

        if total_length == 0:
            raise MisconfigurationException(
                "Total length of `Dataloader` across ranks is zero. Please make sure that it returns at least 1 batch."
            )
        if total_length > 0 and local_length == 0:
            if model.allow_zero_length_dataloader_with_multiple_devices:
                rank_zero_warn(
                    "Total length of `Dataloader` across ranks is zero, but local rank has zero length."
                    " Please be cautious of uneven batch length."
                )
                has_len = False
            else:
                raise MisconfigurationException(
                    "`Dataloader` within local rank has zero length. Please make sure that it returns at least 1 batch."
                )
        else:
            has_len = True

    except TypeError:
        has_len = False
    except NotImplementedError:  # e.g. raised by torchtext if a batch_size_fn is used
        has_len = False

    if has_len and has_iterable_dataset(dataloader):
        rank_zero_warn(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-process data loading (when num_workers > 1),"
            " `__len__` could be inaccurate if each worker is not configured independently"
            " to avoid having duplicate data."
        )
    return has_len


def get_len(dataloader: DataLoader) -> Union[int, float]:
    """Return the length of the given DataLoader.

    If ``__len__`` method is not implemented, return float('inf').
    """

    if has_len(dataloader):
        return len(dataloader)

    return float("inf")
