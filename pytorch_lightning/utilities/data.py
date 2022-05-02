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
import functools
import inspect
import os
from contextlib import contextmanager
from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, Generator, Iterable, Mapping, Optional, Set, Type, Union

import torch
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler, Sampler, SequentialSampler

import pytorch_lightning as pl
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.auto_restart import CaptureIterableDataset, CaptureMapDataset, FastForwardSampler
from pytorch_lightning.utilities.enums import _FaultTolerantMode
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.seed import pl_worker_init_function
from pytorch_lightning.utilities.warnings import WarningCache

BType = Union[torch.Tensor, str, Mapping[Any, "BType"], Iterable["BType"]]

warning_cache = WarningCache()


def _extract_batch_size(batch: BType) -> Generator[int, None, None]:
    if isinstance(batch, torch.Tensor):
        if batch.ndim == 0:
            yield 1
        else:
            yield batch.size(0)
    elif isinstance(batch, (Iterable, Mapping)) and not isinstance(batch, str):
        if isinstance(batch, Mapping):
            batch = batch.values()

        for sample in batch:
            yield from _extract_batch_size(sample)
    else:
        yield None


def extract_batch_size(batch: BType) -> int:
    """Unpack a batch to find a ``torch.Tensor``.

    Returns:
        ``len(tensor)`` when found, or ``1`` when it hits an empty or non iterable.
    """
    error_msg = (
        "We could not infer the batch_size from the batch. Either simplify its structure"
        " or provide the batch_size as `self.log(..., batch_size=batch_size)`."
    )
    batch_size = None
    try:
        for bs in _extract_batch_size(batch):
            if batch_size is None:
                batch_size = bs
            elif batch_size != bs:
                warning_cache.warn(
                    "Trying to infer the `batch_size` from an ambiguous collection. The batch size we"
                    f" found is {batch_size}. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`."
                )
                break
    except RecursionError:
        raise RecursionError(error_msg)

    if batch_size is None:
        raise MisconfigurationException(error_msg)

    return batch_size


def has_iterable_dataset(dataloader: DataLoader) -> bool:
    return hasattr(dataloader, "dataset") and isinstance(dataloader.dataset, IterableDataset)


def has_len(dataloader: Union[DataLoader, Iterable]) -> bool:
    """Checks if a given Dataloader has ``__len__`` method implemented i.e. if it is a finite dataloader or
    infinite dataloader."""
    try:
        # try getting the length
        if len(dataloader) == 0:
            rank_zero_warn(
                f"`{dataloader.__class__.__name__}` returned 0 length. Please make sure this was your intention."
            )
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
    training_type: "pl.Strategy",
    model: Union["pl.LightningModule", "pl.LightningDataModule"],
) -> bool:
    """Checks if a given Dataloader has ``__len__`` method implemented i.e. if it is a finite dataloader or
    infinite dataloader."""
    try:
        local_length = len(dataloader)
        total_length = training_type.reduce(torch.tensor(local_length).to(model.device), reduce_op="sum")

        if total_length == 0:
            rank_zero_warn(
                f"Total length of `{dataloader.__class__.__name__}` across ranks is zero."
                " Please make sure this was your intention."
            )
        if total_length > 0 and local_length == 0:
            if model.allow_zero_length_dataloader_with_multiple_devices:
                rank_zero_warn(
                    f"Total length of `{dataloader.__class__.__name__}` across ranks is zero, but local rank has zero"
                    " length. Please be cautious of uneven batch length."
                )
                has_len = False
            else:
                raise MisconfigurationException(
                    f"`{dataloader.__class__.__name__}` within local rank has zero length."
                    " Please make sure that it returns at least 1 batch."
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


def _update_dataloader(
    dataloader: DataLoader, sampler: Union[Sampler, Iterable], mode: Optional[RunningStage] = None
) -> DataLoader:
    dl_kwargs = _get_dataloader_init_kwargs(dataloader, sampler, mode=mode)
    dl_cls = type(dataloader)
    try:
        dataloader = dl_cls(**dl_kwargs)
    except TypeError as e:
        # improve exception message due to an incorrect implementation of the `DataLoader` where multiple subclass
        # `__init__` arguments map to one `DataLoader.__init__` argument
        import re

        match = re.match(r".*__init__\(\) got multiple values .* '(\w+)'", str(e))
        if not match:
            # an unexpected `TypeError`, continue failure
            raise
        argument = match.groups()[0]
        message = (
            f"The {dl_cls.__name__} `DataLoader` implementation has an error where more than one `__init__` argument"
            f" can be passed to its parent's `{argument}=...` `__init__` argument. This is likely caused by allowing"
            f" passing both a custom argument that will map to the `{argument}` argument as well as `**kwargs`."
            f" `kwargs` should be filtered to make sure they don't contain the `{argument}` key."
        )
        raise MisconfigurationException(message) from e
    return dataloader


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
    if isinstance(dl_kwargs["dataset"], IterableDataset):
        dl_kwargs["batch_sampler"] = None
        dl_kwargs["sampler"] = None
    else:
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

    if _FaultTolerantMode.detect_current_mode().is_automatic:
        dl_kwargs = _apply_fault_tolerant_automatic_capture_dataset_wrapper(dl_kwargs)

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
    fault_tolerant_mode = _FaultTolerantMode.detect_current_mode()
    batch_sampler = getattr(dataloader, "batch_sampler")
    is_predicting = mode == RunningStage.PREDICTING
    # checking the batch sampler type is different than PyTorch default.
    if batch_sampler is not None and (type(batch_sampler) is not BatchSampler or is_predicting):
        batch_sampler = type(batch_sampler)(
            sampler,
            batch_size=batch_sampler.batch_size,
            drop_last=(False if is_predicting else batch_sampler.drop_last),
        )
        if is_predicting:
            batch_sampler = IndexBatchSamplerWrapper(batch_sampler)

        if fault_tolerant_mode.is_automatic:
            fast_forward_sampler = batch_sampler = FastForwardSampler(batch_sampler)
            fast_forward_sampler.setup(dataloader_batch_size=1)

        return {
            "sampler": None,
            "shuffle": False,
            "batch_sampler": batch_sampler,
            "batch_size": 1,
            "drop_last": False,
        }

    if fault_tolerant_mode.is_automatic:
        fast_forward_sampler = sampler = FastForwardSampler(sampler)
        fast_forward_sampler.setup(dataloader_batch_size=dataloader.batch_size)

    return {"sampler": sampler, "shuffle": False, "batch_sampler": None}


def _auto_add_worker_init_fn(dataloader: DataLoader, rank: int) -> None:
    if int(os.environ.get("PL_SEED_WORKERS", 0)) and dataloader.worker_init_fn is None:
        dataloader.worker_init_fn = partial(pl_worker_init_function, rank=rank)


def _wrap_init(init: Callable) -> Callable:
    """Wraps the ``__init__`` method of the dataloader in order to enable re-instantiation of custom subclasses of
    :class:`~torch.utils.data.DataLoader`."""

    @functools.wraps(init)
    def wrapper(obj: DataLoader, *args: Any, **kwargs: Any) -> None:
        # We need to inspect `init`, as inspecting `obj.__init__`
        # can lead to inspecting the wrong function with multiple inheritance
        params = inspect.signature(init).parameters

        param_names = [
            param.name
            for param in params.values()
            if param.name != "self" and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        ]

        cls = type(obj)
        for arg_name, arg_value in chain(zip(param_names, args), kwargs.items()):
            if hasattr(cls, arg_name) and getattr(cls, arg_name).fset is None:
                # the class defines a read-only (no setter) property of this name. it's likely that the implementation
                # will set `self._arg_name = arg_value` in `__init__` which is the attribute returned by the `arg_name`
                # property so we are fine skipping in that case
                continue
            setattr(obj, arg_name, arg_value)
        init(obj, *args, **kwargs)

    return wrapper


# https://stackoverflow.com/a/63851681/9201239
def _get_all_subclasses(cls: Type[Any]) -> Set[Type[Any]]:
    """Returns a list of all classes that inherit directly or indirectly from the given class."""
    subclasses = set()

    def recurse(cl: Type[Any]) -> None:
        for subclass in cl.__subclasses__():
            subclasses.add(subclass)
            recurse(subclass)

    recurse(cls)
    return subclasses


@contextmanager
def _replace_dataloader_init_method() -> Generator[None, None, None]:
    """This context manager is used to add support for re-instantiation of custom (subclasses) of
    :class:`~torch.utils.data.DataLoader`. It patches the ``__init__`` method."""
    subclasses = _get_all_subclasses(DataLoader)
    for subclass in subclasses:
        subclass._old_init = subclass.__init__
        subclass.__init__ = _wrap_init(subclass.__init__)
    yield
    for subclass in subclasses:
        subclass.__init__ = subclass._old_init
        del subclass._old_init


def _apply_fault_tolerant_automatic_capture_dataset_wrapper(dl_kwargs: Dict) -> Dict:
    dataset = dl_kwargs["dataset"]
    if isinstance(dataset, IterableDataset):
        # wrap the `IterableDataset` into a `CaptureIterableDataset` to record sampler states.
        dl_kwargs["dataset"] = CaptureIterableDataset(dataset=dataset)
    elif get_len(dataset) != float("inf"):
        dl_kwargs["dataset"] = CaptureMapDataset(dataset=dataset)
    else:
        raise MisconfigurationException("This shouldn't happen, please open an issue on Lightning Github repository.")
    return dl_kwargs


def _is_dataloader_shuffled(dataloader: object) -> bool:
    if hasattr(dataloader, "shuffle"):
        # this attribute is not part of PyTorch's DataLoader, but could have been set by
        # our `_replace_dataloader_init_method` context manager
        return dataloader.shuffle
    if isinstance(dataloader.dataset, IterableDataset):
        # shuffling is useless with iterable datasets
        return False
    if not hasattr(dataloader, "sampler"):
        # shuffling is enabled via a sampler. No sampler, no shuffling
        return False
    sampler = dataloader.sampler
    if isinstance(sampler, SequentialSampler):
        return False
    return isinstance(sampler, RandomSampler)
