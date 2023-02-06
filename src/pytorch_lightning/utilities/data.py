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
from dataclasses import fields
from typing import Any, Dict, Generator, Iterable, Mapping, Optional, Tuple, Union

import torch
from lightning_utilities.core.apply_func import is_dataclass_instance
from torch import Tensor
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    Sampler,
    SequentialSampler,
)

import pytorch_lightning as pl
from lightning_fabric.utilities.data import _reinstantiate_wrapped_cls, _replace_value_in_saved_args
from lightning_fabric.utilities.data import has_iterable_dataset as new_has_iterable_dataset
from lightning_fabric.utilities.data import has_len as new_has_len
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.trainer.supporters import CombinedLoader
from pytorch_lightning.utilities.auto_restart import CaptureIterableDataset, CaptureMapDataset, FastForwardSampler
from pytorch_lightning.utilities.enums import _FaultTolerantMode
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn, WarningCache

# might be supported in later releases, see https://github.com/python/mypy/pull/13297
BType = Union[Tensor, str, Mapping[Any, "BType"], Iterable["BType"]]  # type: ignore[misc]

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
        for field in fields(batch):
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
    dataloader: Union[DataLoader, CombinedLoader],
    strategy: "pl.strategies.Strategy",
    model: Union["pl.LightningModule", "pl.LightningDataModule"],
) -> bool:
    """Checks if a given Dataloader has ``__len__`` method implemented i.e. if it is a finite dataloader or
    infinite dataloader."""
    try:
        local_length = len(dataloader)  # type: ignore [arg-type] # we are checking with duck-typing
        total_length = strategy.reduce(torch.tensor(local_length, device=strategy.root_device), reduce_op="sum")

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

    except (TypeError, NotImplementedError):
        has_len = False

    # we are checking using lightning_fabric, which doesn't know CombinedLoader
    if has_len and new_has_iterable_dataset(dataloader):  # type: ignore [arg-type]
        rank_zero_warn(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-process data loading (when num_workers > 1),"
            " `__len__` could be inaccurate if each worker is not configured independently"
            " to avoid having duplicate data."
        )
    return has_len


def get_len(dataloader: Union[DataLoader, Dataset]) -> Union[int, float]:
    """Return the length of the given DataLoader.

    If ``__len__`` method is not implemented, return float('inf').
    """

    if new_has_len(dataloader):
        return len(dataloader)  # type: ignore [arg-type]

    return float("inf")


def _update_dataloader(
    dataloader: DataLoader, sampler: Union[Sampler, Iterable], mode: Optional[RunningStage] = None
) -> DataLoader:
    dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, sampler, mode)
    dataloader = _reinstantiate_wrapped_cls(dataloader, *dl_args, **dl_kwargs)
    return dataloader


def _get_dataloader_init_args_and_kwargs(
    dataloader: DataLoader,
    sampler: Union[Sampler, Iterable],
    mode: Optional[RunningStage] = None,
    disallow_batch_sampler: bool = False,
) -> Tuple[Tuple[Any], Dict[str, Any]]:
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
            params.update(
                {k: v for k, v in inspect.signature(DataLoader.__init__).parameters.items() if v.default is not v.empty}
            )
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
        dl_kwargs.update(_dataloader_init_kwargs_resolve_sampler(dataloader, sampler, mode, disallow_batch_sampler))

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

    if _FaultTolerantMode.detect_current_mode().is_automatic:
        dl_args, dl_kwargs = _apply_fault_tolerant_automatic_capture_dataset_wrapper(
            was_wrapped, arg_names, dl_args, dl_kwargs
        )

    return dl_args, dl_kwargs


def _dataloader_init_kwargs_resolve_sampler(
    dataloader: DataLoader,
    sampler: Union[Sampler, Iterable],
    mode: Optional[RunningStage] = None,
    disallow_batch_sampler: bool = False,
) -> Dict[str, Any]:
    """This function is used to handle the sampler, batch_sampler arguments associated within a DataLoader for its
    re-instantiation.

    If the dataloader is being used for prediction, the sampler will be wrapped into an `IndexBatchSamplerWrapper`, so
    Lightning can keep track of its indices. If fault tolerant training is enabled, the sampler will be wrapped into a
    `FastForwardSampler`.

    If there are multiple devices in IPU mode, it is necessary to disallow BatchSampler that isn't instantiated
    automatically, since `poptorch.DataLoader` will try to increase the batch_size
    """
    fault_tolerant_mode = _FaultTolerantMode.detect_current_mode()
    batch_sampler = getattr(dataloader, "batch_sampler")
    is_predicting = mode == RunningStage.PREDICTING

    if batch_sampler is not None:
        if disallow_batch_sampler:
            # Check that we don't have a PyTorch default batch sampler that was instantiated in DataLoader __init__
            if not (
                type(batch_sampler) is BatchSampler
                and batch_sampler.sampler == sampler
                and dataloader.batch_size == batch_sampler.batch_size
            ):
                raise MisconfigurationException(
                    "It is not possible to have a batch sampler in your dataloader, "
                    "when running on multiple IPU devices."
                )
        elif type(batch_sampler) is not BatchSampler or is_predicting:
            batch_sampler_cls = type(batch_sampler)
            if hasattr(batch_sampler, "__pl_saved_args"):
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
            else:
                try:
                    batch_sampler = batch_sampler_cls(
                        sampler,
                        batch_size=batch_sampler.batch_size,
                        drop_last=(False if is_predicting else batch_sampler.drop_last),
                    )
                except TypeError as e:
                    import re

                    match = re.match(r".*__init__\(\) (got multiple values)|(missing \d required)", str(e))
                    if not match:
                        # an unexpected `TypeError`, continue failure
                        raise

                    # There could either be too few or too many arguments. Customizing the message based on this doesn't
                    # make much sense since our MisconfigurationException is going to be raised from the original one.
                    raise MisconfigurationException(
                        "We tried to re-instantiate your custom batch sampler and failed. "
                        "To mitigate this, either follow the API of `BatchSampler` or instantiate "
                        "your custom batch sampler inside `*_dataloader` hooks of your module."
                    ) from e

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


def _wrap_with_capture_dataset(dataset: Dataset) -> Dataset:
    if isinstance(dataset, IterableDataset):
        # wrap the `IterableDataset` into a `CaptureIterableDataset` to record sampler states.
        return CaptureIterableDataset(dataset=dataset)
    if get_len(dataset) != float("inf"):
        return CaptureMapDataset(dataset=dataset)
    raise RuntimeError("This shouldn't happen, please open an issue on Lightning Github repository.")


def _apply_fault_tolerant_automatic_capture_dataset_wrapper(
    was_wrapped: bool, arg_names: Tuple[str, ...], dl_args: Tuple[Any, ...], dl_kwargs: Dict[str, Any]
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    if "dataset" in dl_kwargs:
        dl_kwargs["dataset"] = _wrap_with_capture_dataset(dl_kwargs["dataset"])
    elif "dataset" in arg_names:
        dataset_idx = arg_names.index("dataset")
        dataset = _wrap_with_capture_dataset(dl_args[dataset_idx])
        dl_args = dl_args[:dataset_idx] + (dataset,) + dl_args[dataset_idx + 1 :]
    else:
        if was_wrapped:
            avoid_message = (
                " To avoid this, either pass `DataLoader(dataset=your_dataset)` or the positional dataset argument"
                " `DataLoader(your_dataset, ...)`."
            )
        else:
            avoid_message = " To avoid this, define `self.dataset = dataset` inside your DataLoader's `__init__`."

        raise MisconfigurationException(
            "You enabled automatic Fault Tolerant mode, but we were not able to replace your dataset"
            " with Fault Tolerant wrapper, because you have a custom DataLoader." + avoid_message
        )

    return dl_args, dl_kwargs


def _is_dataloader_shuffled(dataloader: object) -> bool:
    if hasattr(dataloader, "__pl_saved_kwargs"):
        # this attribute is not part of PyTorch's DataLoader, but could have been set by
        # our `_replace_init_method` context manager
        if "shuffle" in dataloader.__pl_saved_kwargs:
            return dataloader.__pl_saved_kwargs["shuffle"]
        if "shuffle" in dataloader.__pl_saved_arg_names:
            return dataloader.__pl_saved_args[dataloader.__pl_saved_arg_names.index("shuffle")]
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


def has_iterable_dataset(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.data.has_iterable_dataset` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_fabric.utilities.data.has_iterable_dataset` instead."
    )
    return new_has_iterable_dataset(*args, **kwargs)


def has_len(*args: Any, **kwargs: Any) -> Any:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.data.has_len` has been deprecated in v1.8.0 and will be"
        " removed in v2.0.0. Please use `lightning_fabric.utilities.data.has_len` instead."
    )
    return new_has_len(*args, **kwargs)
