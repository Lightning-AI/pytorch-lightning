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
import inspect
from collections.abc import Generator, Iterable, Mapping, Sized
from dataclasses import fields
from typing import Any, Optional, Union

import torch
from lightning_utilities.core.apply_func import is_dataclass_instance
from torch import Tensor
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler, Sampler, SequentialSampler
from typing_extensions import TypeGuard

import lightning.pytorch as pl
from lightning.fabric.utilities.data import (
    _reinstantiate_wrapped_cls,
    _replace_value_in_saved_args,
    has_iterable_dataset,
    sized_len,
)
from lightning.fabric.utilities.warnings import PossibleUserWarning
from lightning.pytorch.overrides.distributed import _IndexBatchSamplerWrapper
from lightning.pytorch.trainer.states import RunningStage
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from lightning.pytorch.utilities.rank_zero import WarningCache, rank_zero_warn

BType = Union[Tensor, str, Mapping[Any, "BType"], Iterable["BType"]]

warning_cache = WarningCache()


def _extract_batch_size(batch: BType) -> Generator[Optional[int], None, None]:
    if isinstance(batch, Tensor):
        if batch.ndim == 0:
            yield 1
        else:
            yield batch.size(0)
    elif isinstance(batch, (Iterable, Mapping)) and not isinstance(batch, str):
        if isinstance(batch, Mapping):
            batch = batch.values()

        for sample in batch:
            yield from _extract_batch_size(sample)
    elif is_dataclass_instance(batch):
        for field in fields(batch):  # type: ignore[arg-type]
            yield from _extract_batch_size(getattr(batch, field.name))
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


def has_len_all_ranks(
    dataloader: object,
    strategy: "pl.strategies.Strategy",
    allow_zero_length_dataloader_with_multiple_devices: bool = False,
) -> TypeGuard[Sized]:
    """Checks if a given object has ``__len__`` method implemented on all ranks."""
    local_length = sized_len(dataloader)
    if local_length is None:
        # __len__ is not defined, skip these checks
        return False

    total_length = strategy.reduce(torch.tensor(local_length, device=strategy.root_device), reduce_op="sum")
    if total_length == 0:
        rank_zero_warn(
            f"Total length of `{type(dataloader).__name__}` across ranks is zero."
            " Please make sure this was your intention."
        )
    if total_length > 0 and local_length == 0:
        dataloader_cls_name = type(dataloader).__name__
        if not allow_zero_length_dataloader_with_multiple_devices:
            raise RuntimeError(
                f"`{dataloader_cls_name}` within local rank has zero length."
                " Please make sure that it returns at least 1 batch."
            )
        rank_zero_warn(
            f"Total length of `{dataloader_cls_name}` across ranks is zero, but local rank has zero"
            " length. Please be cautious of uneven batch length."
        )

    if has_iterable_dataset(dataloader):
        rank_zero_warn(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-process data loading (when num_workers > 1),"
            " `__len__` could be inaccurate if each worker is not configured independently"
            " to avoid having duplicate data."
        )
    return True


def _update_dataloader(
    dataloader: DataLoader, sampler: Union[Sampler, Iterable], mode: Optional[RunningStage] = None
) -> DataLoader:
    dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, sampler, mode)
    return _reinstantiate_wrapped_cls(dataloader, *dl_args, **dl_kwargs)


def _get_dataloader_init_args_and_kwargs(
    dataloader: DataLoader,
    sampler: Union[Sampler, Iterable],
    mode: Optional[RunningStage] = None,
) -> tuple[tuple[Any], dict[str, Any]]:
    if not isinstance(dataloader, DataLoader):
        raise ValueError(f"The dataloader {dataloader} needs to subclass `torch.utils.data.DataLoader`")

    was_wrapped = hasattr(dataloader, "__pl_saved_args")
    if was_wrapped:
        dl_args = dataloader.__pl_saved_args
        dl_kwargs = dataloader.__pl_saved_kwargs
        arg_names = dataloader.__pl_saved_arg_names
        original_dataset = dataloader.__dataset  # we have this saved from _wrap_init
    else:
        # get the dataloader instance attributes
        attrs = {k: v for k, v in vars(dataloader).items() if not k.startswith("_")}
        # We cannot be 100% sure the class sets dataset argument. Let's set it to None to be safe
        # and hope we can get it from the instance attributes
        original_dataset = None
        # not part of `vars`
        attrs["multiprocessing_context"] = dataloader.multiprocessing_context
        arg_names = ()

    # get the dataloader instance `__init__` parameters
    params = dict(inspect.signature(dataloader.__init__).parameters)  # type: ignore[misc]
    has_variadic_kwargs = any(p.kind is p.VAR_KEYWORD for p in params.values())
    if has_variadic_kwargs:
        # if the signature takes **kwargs, assume they will be passed down with `super().__init__(**kwargs)`

        if was_wrapped:
            # if the dataloader was wrapped in a hook, only take arguments with default values
            # and assume user passes their kwargs correctly
            params.update({
                k: v for k, v in inspect.signature(DataLoader.__init__).parameters.items() if v.default is not v.empty
            })
        else:
            params.update(inspect.signature(DataLoader.__init__).parameters)
            params.pop("self", None)

    if not was_wrapped:
        # keep only the params whose default is different to the current attr value
        non_defaults = {name for name, p in params.items() if name in attrs and p.default is not attrs[name]}

        # add `dataset` as it might have been replaced with `*args`
        non_defaults.add("dataset")
        # kwargs to re-construct the dataloader
        dl_kwargs = {k: v for k, v in attrs.items() if k in non_defaults}
        dl_args = ()

    dataset = dl_kwargs.get("dataset", original_dataset)
    if isinstance(dataset, IterableDataset):
        dl_kwargs["batch_sampler"] = None
        dl_kwargs["sampler"] = None
    else:
        dl_kwargs.update(_dataloader_init_kwargs_resolve_sampler(dataloader, sampler, mode))

    required_args = {
        p.name
        for p in params.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        and p.default is p.empty
        and p.name not in dl_kwargs
        and p.name not in arg_names
    }
    # the dataloader has required args which we could not extract from the existing attributes
    if required_args:
        sorted_required_args = sorted(required_args)
        dataloader_cls_name = dataloader.__class__.__name__
        missing_args_message = ", ".join(f"`self.{arg_name}`" for arg_name in sorted_required_args)
        raise MisconfigurationException(
            f"Trying to inject custom `Sampler` into the `{dataloader_cls_name}` instance. "
            "This would fail as some of the `__init__` arguments are not available as instance attributes. "
            f"The missing attributes are {sorted_required_args}. If you instantiate your `{dataloader_cls_name}` "
            "inside a `*_dataloader` hook of your module, we will do this for you."
            f" Otherwise, define {missing_args_message} inside your `__init__`."
        )

    if not has_variadic_kwargs:
        # the dataloader signature does not allow keyword arguments that need to be passed
        missing_kwargs = (set(dl_kwargs) | set(arg_names)) - params.keys()
        if missing_kwargs:
            sorted_missing_kwargs = sorted(missing_kwargs)
            dataloader_cls_name = dataloader.__class__.__name__
            raise MisconfigurationException(
                f"Trying to inject parameters into the `{dataloader_cls_name}` instance. "
                "This would fail as it doesn't expose all its attributes in the `__init__` signature. "
                f"The missing arguments are {sorted_missing_kwargs}. HINT: If you wrote the `{dataloader_cls_name}` "
                "class, add the `__init__` arguments or allow passing `**kwargs`"
            )

    return dl_args, dl_kwargs


def _dataloader_init_kwargs_resolve_sampler(
    dataloader: DataLoader,
    sampler: Union[Sampler, Iterable],
    mode: Optional[RunningStage] = None,
) -> dict[str, Any]:
    """This function is used to handle the sampler, batch_sampler arguments associated within a DataLoader for its re-
    instantiation.

    If the dataloader is being used for prediction, the sampler will be wrapped into an `_IndexBatchSamplerWrapper`, so
    Lightning can keep track of its indices.

    """
    is_predicting = mode == RunningStage.PREDICTING
    batch_sampler = getattr(dataloader, "batch_sampler")
    batch_sampler_cls = type(batch_sampler)

    if batch_sampler is not None and (batch_sampler_cls is not BatchSampler or is_predicting):
        if hasattr(batch_sampler, "__pl_saved_args"):
            # This is a PyTorch `BatchSampler` subclass for which we captured the init args
            args = batch_sampler.__pl_saved_args
            kwargs = batch_sampler.__pl_saved_kwargs
            default_kwargs = batch_sampler.__pl_saved_default_kwargs
            arg_names = batch_sampler.__pl_saved_arg_names

            if is_predicting:
                success, args, kwargs = _replace_value_in_saved_args(
                    "drop_last", False, args, kwargs, default_kwargs, arg_names
                )
                if not success:
                    rank_zero_warn(
                        f"Trying to inject `drop_last=False` into batch sampler since you are predicting, however "
                        f"it seems the class `{batch_sampler_cls.__qualname__}` does not support it. "
                        "Your predictions might be incomplete. To mitigate this, expose `drop_last` in "
                        "the `__init__` method of your custom class."
                    )

            success, args, kwargs = _replace_value_in_saved_args(
                "sampler", sampler, args, kwargs, default_kwargs, arg_names
            )
            if not success:
                raise TypeError(
                    "Trying to inject a modified sampler into the batch sampler; however, it seems the class "
                    f"`{batch_sampler_cls.__qualname__}` does not have an argument called `sampler.` To mitigate "
                    "this, expose an argument `sampler` in the `__init__` method of your custom class."
                )

            batch_sampler = _reinstantiate_wrapped_cls(batch_sampler, *args, **kwargs)
        elif hasattr(batch_sampler, "batch_size") and hasattr(batch_sampler, "drop_last"):
            # This is a sampler for which we could not capture the init args, but it kinda looks like a batch sampler
            # even if it does not inherit from PyTorch's interface.
            try:
                batch_sampler = batch_sampler_cls(
                    sampler,
                    batch_size=batch_sampler.batch_size,
                    drop_last=(False if is_predicting else batch_sampler.drop_last),
                )
            except TypeError as ex:
                import re

                match = re.match(r".*__init__\(\) (got multiple values)|(missing \d required)", str(ex))
                if not match:
                    # an unexpected `TypeError`, continue failure
                    raise

                # There could either be too few or too many arguments. Customizing the message based on this doesn't
                # make much sense since our MisconfigurationException is going to be raised from the original one.
                raise TypeError(
                    " Lightning can't inject a (distributed) sampler into your batch sampler, because it doesn't"
                    " subclass PyTorch's `BatchSampler`. To mitigate this, either follow the API of `BatchSampler` and"
                    " instantiate your custom batch sampler inside the `*_dataloader` hook of your module,"
                    " or set `Trainer(use_distributed_sampler=False)`. If you choose the latter, you will be"
                    " responsible for handling the distributed sampling within your batch sampler."
                ) from ex
        elif is_predicting:
            rank_zero_warn(
                f"You are using a custom batch sampler `{batch_sampler_cls.__qualname__}` for prediction."
                " Lightning would normally set `drop_last=False` to ensure all samples are returned, but for"
                " custom samplers it can't guarantee this. Make sure your sampler is configured correctly to return"
                " all indices.",
                category=PossibleUserWarning,
            )
        else:
            # The sampler is not a PyTorch `BatchSampler`, we don't know how to inject a custom sampler or
            # how to adjust the `drop_last` value
            raise TypeError(
                " Lightning can't inject a (distributed) sampler into your batch sampler, because it doesn't"
                " subclass PyTorch's `BatchSampler`. To mitigate this, either follow the API of `BatchSampler`"
                " or set `Trainer(use_distributed_sampler=False)`. If you choose the latter, you will be"
                " responsible for handling the distributed sampling within your batch sampler."
            )

        if is_predicting:
            batch_sampler = _IndexBatchSamplerWrapper(batch_sampler)

        # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
        return {
            "sampler": None,
            "shuffle": False,
            "batch_sampler": batch_sampler,
            "batch_size": 1,
            "drop_last": False,
        }

    return {"sampler": sampler, "shuffle": False, "batch_sampler": None}


def _is_dataloader_shuffled(dataloader: object) -> bool:
    if hasattr(dataloader, "__pl_saved_kwargs"):
        # this attribute is not part of PyTorch's DataLoader, but could have been set by
        # our `_replace_init_method` context manager
        if "shuffle" in dataloader.__pl_saved_kwargs:
            return dataloader.__pl_saved_kwargs["shuffle"]
        if "shuffle" in dataloader.__pl_saved_arg_names:
            return dataloader.__pl_saved_args[dataloader.__pl_saved_arg_names.index("shuffle")]
    if hasattr(dataloader, "dataset") and isinstance(dataloader.dataset, IterableDataset):
        # shuffling is useless with iterable datasets
        return False
    if not hasattr(dataloader, "sampler"):
        # shuffling is enabled via a sampler. No sampler, no shuffling
        return False
    batch_sampler = dataloader.batch_sampler
    if batch_sampler is not None:
        # custom batch samplers may not have an internal .sampler
        sampler = batch_sampler.sampler if hasattr(batch_sampler, "sampler") else batch_sampler
    else:
        sampler = dataloader.sampler
    if isinstance(sampler, SequentialSampler):
        return False
    return isinstance(sampler, RandomSampler)
