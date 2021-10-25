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
import inspect
from typing import Any, Dict, Iterable, Mapping, Optional, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, Sampler

from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities import rank_zero_warn
from pytorch_lightning.utilities.auto_restart import CaptureIterableDataset, CaptureMapDataset, FastForwardSampler
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _fault_tolerant_training

BType = Union[torch.Tensor, str, Mapping[Any, "BType"], Iterable["BType"]]


def extract_batch_size(batch: BType) -> int:
    """Recursively unpack a batch to find a torch.Tensor.

    Returns:
        ``len(tensor)`` when found, or ``1`` when it hits an empty or non iterable.
    """
    if isinstance(batch, torch.Tensor):
        return batch.size(0)
    if isinstance(batch, str):
        return len(batch)
    if isinstance(batch, dict):
        sample = next(iter(batch.values()), 1)
        return extract_batch_size(sample)
    if isinstance(batch, Iterable):
        sample = next(iter(batch), 1)
        return extract_batch_size(sample)

    return 1


def has_iterable_dataset(dataloader: DataLoader) -> bool:
    return hasattr(dataloader, "dataset") and isinstance(dataloader.dataset, IterableDataset)


def has_len(dataloader: DataLoader) -> bool:
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


def get_len(dataloader: DataLoader) -> Union[int, float]:
    """Return the length of the given DataLoader.

    If ``__len__`` method is not implemented, return float('inf').
    """

    if has_len(dataloader):
        return len(dataloader)

    return float("inf")


def _get_dataloader_init_kwargs(
    dataloader: DataLoader, sampler: Optional[Sampler], mode: Optional[RunningStage] = None
) -> Dict[str, Any]:
    if not isinstance(dataloader, DataLoader):
        raise ValueError(f"The dataloader {dataloader} needs to subclass `torch.utils.data.DataLoader`")

    # get the dataloader instance attributes
    attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}
    # not part of `vars`
    attrs["multiprocessing_context"] = dataloader.multiprocessing_context

    # get the dataloader instance `__init__` parameters
    params = dict(inspect.signature(dataloader.__init__).parameters)
    has_variadic_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    if has_variadic_kwargs:
        # if the signature takes **kwargs, assume they will be passed down with `super().__init__(**kwargs)`
        params.update(inspect.signature(DataLoader.__init__).parameters)
        del params["self"]

    # keep only the params whose default is different to the current attr value
    non_defaults = {name for name, p in params.items() if name in attrs and p.default != attrs[name]}
    # add `dataset` as it might have been replaced with `*args`
    non_defaults.add("dataset")

    # kwargs to re-construct the dataloader
    dl_kwargs = {k: v for k, v in attrs.items() if k in non_defaults}
    dl_kwargs.update(_dataloader_init_kwargs_resolve_sampler(dataloader, sampler, mode=mode))

    required_args = {
        p.name
        for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty and p.name not in dl_kwargs
    }
    # the dataloader has required args which we could not extract from the existing attributes
    if required_args:
        required_args = sorted(required_args)
        dataloader_cls_name = dataloader.__class__.__name__
        raise MisconfigurationException(
            f"Trying to inject `DistributedSampler` into the `{dataloader_cls_name}` instance. "
            "This would fail as some of the `__init__` arguments are not available as instance attributes. "
            f"The missing attributes are {required_args}. "
            f"HINT: If you wrote the `{dataloader_cls_name}` class, define `self.missing_arg_name` or "
            "manually add the `DistributedSampler` as: "
            f"`{dataloader_cls_name}(dataset, sampler=DistributedSampler(dataset))`."
        )

    if not has_variadic_kwargs:
        # the dataloader signature does not allow keyword arguments that need to be passed
        missing_kwargs = dl_kwargs.keys() - params.keys()
        if missing_kwargs:
            missing_kwargs = sorted(missing_kwargs)
            dataloader_cls_name = dataloader.__class__.__name__
            raise MisconfigurationException(
                f"Trying to inject `DistributedSampler` into the `{dataloader_cls_name}` instance. "
                "This would fail as it doesn't expose all its attributes in the `__init__` signature. "
                f"The missing arguments are {missing_kwargs}. "
                f"HINT: If you wrote the `{dataloader_cls_name}` class, add the `__init__` arguments or "
                "manually add the `DistributedSampler` as: "
                f"`{dataloader_cls_name}(dataset, sampler=DistributedSampler(dataset))`."
            )

    if isinstance(dl_kwargs["dataset"], IterableDataset):
        dl_kwargs["batch_sampler"] = None
        dl_kwargs["sampler"] = None

    if _fault_tolerant_training():
        if isinstance(dl_kwargs["dataset"], IterableDataset):
            # wrap the `IterableDataset` into a `CaptureIterableDataset` to record sampler states.
            dl_kwargs["dataset"] = CaptureIterableDataset(dataset=dl_kwargs["dataset"])
        elif len(dl_kwargs["dataset"]):
            dl_kwargs["dataset"] = CaptureMapDataset(dataset=dl_kwargs["dataset"])
        else:
            raise MisconfigurationException(
                "This shouldn't happen, please open an issue on Lightning Github repository."
            )

    return dl_kwargs


def _dataloader_init_kwargs_resolve_sampler(
    dataloader: DataLoader, sampler: Optional[Sampler], mode: Optional[RunningStage] = None
) -> Dict[str, Any]:
    """This function is used to handle the sampler, batch_sampler arguments associated within a DataLoader for its
    re-instantiation.

    If the dataloader is being used for prediction, the sampler will be wrapped into an `IndexBatchSamplerWrapper`, so
    Lightning can keep track of its indices. If fault tolerant training is enabled, the sampler will be wrapped into a
    `FastForwardSampler`.
    """
    batch_sampler = getattr(dataloader, "batch_sampler")
    is_predicting = mode == RunningStage.PREDICTING
    # checking the batch sampler type is different than PyTorch default.
    if (batch_sampler is not None and type(batch_sampler) is not BatchSampler) or is_predicting:
        batch_sampler = type(batch_sampler)(
            sampler,
            batch_size=batch_sampler.batch_size,
            drop_last=(False if is_predicting else batch_sampler.drop_last),
        )
        if is_predicting:
            batch_sampler = IndexBatchSamplerWrapper(batch_sampler)

        if _fault_tolerant_training():
            fast_forward_sampler = batch_sampler = FastForwardSampler(batch_sampler)
            fast_forward_sampler.setup(dataloader_batch_size=1)

        return {
            "sampler": None,
            "shuffle": False,
            "batch_sampler": batch_sampler,
            "batch_size": 1,
            "drop_last": False,
        }

    if _fault_tolerant_training():
        fast_forward_sampler = sampler = FastForwardSampler(sampler)
        fast_forward_sampler.setup(dataloader_batch_size=dataloader.batch_size)

    return {"sampler": sampler, "shuffle": False, "batch_sampler": None}
