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
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
from functools import partial
from typing import Any, Callable, Dict, Generator, Iterable, Mapping, Optional, Set, Tuple, Type, Union

import torch
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
from pytorch_lightning.overrides.distributed import IndexBatchSamplerWrapper
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.apply_func import _is_dataclass_instance
from pytorch_lightning.utilities.auto_restart import CaptureIterableDataset, CaptureMapDataset, FastForwardSampler
from pytorch_lightning.utilities.enums import _FaultTolerantMode, LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.seed import pl_worker_init_function
from pytorch_lightning.utilities.warnings import WarningCache

BType = Union[Tensor, str, Mapping[Any, "BType"], Iterable["BType"]]

warning_cache = WarningCache()


class _WrapAttrTag(LightningEnum):
    SET = "set"
    DEL = "del"

    def __call__(self, *args):
        if self == self.SET:
            fn = setattr
        else:
            fn = delattr
        return fn(*args)


def _extract_batch_size(batch: BType) -> Generator[int, None, None]:
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
    elif _is_dataclass_instance(batch):
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
    dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, sampler, mode)
    dataloader = _reinstantiate_wrapped_cls(dataloader, *dl_args, **dl_kwargs)
    return dataloader


def _get_dataloader_init_args_and_kwargs(
    dataloader: DataLoader,
    sampler: Optional[Sampler],
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
    params = dict(inspect.signature(dataloader.__init__).parameters)
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
        non_defaults = {name for name, p in params.items() if name in attrs and p.default != attrs[name]}

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
        required_args = sorted(required_args)
        dataloader_cls_name = dataloader.__class__.__name__
        missing_args_message = ", ".join(f"`self.{arg_name}`" for arg_name in required_args)
        raise MisconfigurationException(
            f"Trying to inject custom `Sampler` into the `{dataloader_cls_name}` instance. "
            "This would fail as some of the `__init__` arguments are not available as instance attributes. "
            f"The missing attributes are {required_args}. If you instantiate your `{dataloader_cls_name}` inside a "
            "`*_dataloader` hook of your module, we will do this for you."
            f" Otherwise, define {missing_args_message} inside your `__init__`."
        )

    if not has_variadic_kwargs:
        # the dataloader signature does not allow keyword arguments that need to be passed
        missing_kwargs = (set(dl_kwargs) | set(arg_names)) - params.keys()
        if missing_kwargs:
            missing_kwargs = sorted(missing_kwargs)
            dataloader_cls_name = dataloader.__class__.__name__
            raise MisconfigurationException(
                f"Trying to inject parameters into the `{dataloader_cls_name}` instance. "
                "This would fail as it doesn't expose all its attributes in the `__init__` signature. "
                f"The missing arguments are {missing_kwargs}. HINT: If you wrote the `{dataloader_cls_name}` class, "
                "add the `__init__` arguments or allow passing `**kwargs`"
            )

    if _FaultTolerantMode.detect_current_mode().is_automatic:
        dl_args, dl_kwargs = _apply_fault_tolerant_automatic_capture_dataset_wrapper(
            was_wrapped, arg_names, dl_args, dl_kwargs
        )

    return dl_args, dl_kwargs


def _dataloader_init_kwargs_resolve_sampler(
    dataloader: DataLoader,
    sampler: Optional[Sampler],
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


def _replace_value_in_saved_args(
    replace_key: str,
    replace_value: Any,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    default_kwargs: Dict[str, Any],
    arg_names: Tuple[str, ...],
) -> Tuple[bool, Tuple[Any, ...], Dict[str, Any]]:
    """Tries to replace an argument value in a saved list of args and kwargs.

    Returns a tuple indicating success of the operation and modified saved args and kwargs
    """

    if replace_key in arg_names:
        replace_index = arg_names.index(replace_key)
        args = args[:replace_index] + (replace_value,) + args[replace_index + 1 :]
        return True, args, kwargs
    elif replace_key in kwargs or replace_key in default_kwargs:
        kwargs[replace_key] = replace_value
        return True, args, kwargs

    return False, args, kwargs


def _auto_add_worker_init_fn(dataloader: DataLoader, rank: int) -> None:
    if int(os.environ.get("PL_SEED_WORKERS", 0)) and dataloader.worker_init_fn is None:
        dataloader.worker_init_fn = partial(pl_worker_init_function, rank=rank)


def _reinstantiate_wrapped_cls(orig_object: Any, *args: Any, explicit_cls: Optional[Type] = None, **kwargs: Any) -> Any:
    constructor = type(orig_object) if explicit_cls is None else explicit_cls

    try:
        result = constructor(*args, **kwargs)
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
            f"The {constructor.__name__} implementation has an error where more than one `__init__` argument"
            f" can be passed to its parent's `{argument}=...` `__init__` argument. This is likely caused by allowing"
            f" passing both a custom argument that will map to the `{argument}` argument as well as `**kwargs`."
            f" `kwargs` should be filtered to make sure they don't contain the `{argument}` key."
            " This argument was automatically passed to your object by PyTorch Lightning."
        )
        raise MisconfigurationException(message) from e

    attrs_record = getattr(orig_object, "__pl_attrs_record", list())
    for args, fn in attrs_record:
        fn(result, *args)

    return result


def _wrap_init_method(init: Callable, store_explicit_arg: Optional[str] = None) -> Callable:
    """Wraps the ``__init__`` method of classes (currently :class:`~torch.utils.data.DataLoader` and
    :class:`~torch.utils.data.BatchSampler`) in order to enable re-instantiation of custom subclasses."""

    @functools.wraps(init)
    def wrapper(obj: Any, *args: Any, **kwargs: Any) -> None:
        # We need to inspect `init`, as inspecting `obj.__init__`
        # can lead to inspecting the wrong function with multiple inheritance
        old_inside_init = getattr(obj, "__pl_inside_init", False)
        object.__setattr__(obj, "__pl_inside_init", True)
        params = inspect.signature(init).parameters

        parameters_defaults = OrderedDict(
            (param.name, param.default)
            for param in params.values()
            if param.name != "self" and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        )

        param_names = tuple(parameters_defaults)[: len(args)]

        default_kwargs = {
            name: value
            for name, value in parameters_defaults.items()
            if name not in kwargs and name not in param_names and value != inspect.Parameter.empty
        }

        if not hasattr(obj, "__pl_saved_args"):
            object.__setattr__(obj, "__pl_saved_args", args)
            object.__setattr__(obj, "__pl_saved_kwargs", kwargs)
            object.__setattr__(obj, "__pl_saved_arg_names", param_names)
            object.__setattr__(obj, "__pl_saved_default_kwargs", default_kwargs)

        # We want to use the latest possible value for explicit argument (i.e. ideally what gets passed to base class)
        # so that we can be sure, that it will not get changed anymore.
        # That is why we are setting this in every `__init__`
        if store_explicit_arg is not None:
            if store_explicit_arg in param_names:
                object.__setattr__(obj, f"__{store_explicit_arg}", args[param_names.index(store_explicit_arg)])
            elif store_explicit_arg in kwargs:
                object.__setattr__(obj, f"__{store_explicit_arg}", kwargs[store_explicit_arg])

        init(obj, *args, **kwargs)
        object.__setattr__(obj, "__pl_inside_init", old_inside_init)

    return wrapper


def _wrap_attr_method(method: Callable, tag: _WrapAttrTag) -> Callable:
    """Wraps the ``__setattr__`` or ``__delattr__`` method of classes (currently :class:`~torch.utils.data.DataLoader` and
    :class:`~torch.utils.data.BatchSampler`) in order to enable re-instantiation of custom subclasses."""

    @functools.wraps(method)
    def wrapper(obj: Any, *args: Any):
        # First, let's find out if we're the first in inheritance chain calling the patched method.
        name, *_ = args
        prev_call_name, prev_call_method = getattr(obj, "__pl_current_call", (None, "method"))
        first_call = not (prev_call_name == name and prev_call_method == tag)

        # Then mark the current called method
        object.__setattr__(obj, "__pl_current_call", (name, tag))

        # call original method
        method(obj, *args)
        if first_call and not getattr(obj, "__pl_inside_init", True):
            # and save the value it was called with to the internal list,
            # if we're outside of __init__ and the original call did not fail and we're the first call
            attrs_record = getattr(obj, "__pl_attrs_record", list())
            attrs_record.append((args, tag))
            object.__setattr__(obj, "__pl_attrs_record", attrs_record)
        object.__setattr__(obj, "__pl_current_call", (prev_call_name, prev_call_method))

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
def _replace_dunder_methods(base_cls: Type, store_explicit_arg: Optional[str] = None) -> Generator[None, None, None]:
    """This context manager is used to add support for re-instantiation of custom (subclasses) of `base_cls`.

    It patches the ``__init__``, ``__setattr__`` and ``__delattr__`` methods.
    """
    classes = _get_all_subclasses(base_cls) | {base_cls}
    for cls in classes:
        # Check that __init__ belongs to the class
        # https://stackoverflow.com/a/5253424
        if "__init__" in cls.__dict__:
            cls.__old__init__ = cls.__init__
            cls.__init__ = _wrap_init_method(cls.__init__, store_explicit_arg)

        # we want at least one setattr/delattr in the chain to be patched and it can happen, that none of the subclasses
        # implement `__setattr__`/`__delattr__`. Therefore, we are always patching the `base_cls`
        for patch_fn_name, tag in (("__setattr__", _WrapAttrTag.SET), ("__delattr__", _WrapAttrTag.DEL)):
            if patch_fn_name in cls.__dict__ or cls is base_cls:
                saved_name = f"__old{patch_fn_name}"
                setattr(cls, saved_name, getattr(cls, patch_fn_name))
                setattr(cls, patch_fn_name, _wrap_attr_method(getattr(cls, patch_fn_name), tag))
    yield
    for cls in classes:
        for patched_name in ("__setattr__", "__delattr__", "__init__"):
            # Check that __old__{init,setattr,delattr} belongs to the class
            # https://stackoverflow.com/a/5253424
            if f"__old{patched_name}" in cls.__dict__:
                setattr(cls, patched_name, getattr(cls, f"__old{patched_name}"))
                delattr(cls, f"__old{patched_name}")


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
