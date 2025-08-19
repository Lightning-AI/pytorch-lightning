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

import functools
import inspect
import os
from collections import OrderedDict
from collections.abc import Generator, Iterable, Sized
from contextlib import contextmanager
from functools import partial
from typing import Any, Callable, Optional, Union

from lightning_utilities.core.inheritance import get_all_subclasses
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, Sampler
from typing_extensions import TypeGuard

from lightning.fabric.utilities.enums import LightningEnum
from lightning.fabric.utilities.exceptions import MisconfigurationException
from lightning.fabric.utilities.rank_zero import rank_zero_warn
from lightning.fabric.utilities.seed import pl_worker_init_function


class _WrapAttrTag(LightningEnum):
    SET = "set"
    DEL = "del"

    def __call__(self, *args: Any) -> None:
        fn: Union[Callable[[object, str], None], Callable[[object, str, Any], None]]
        fn = setattr if self == self.SET else delattr
        return fn(*args)


def has_iterable_dataset(dataloader: object) -> bool:
    return hasattr(dataloader, "dataset") and isinstance(dataloader.dataset, IterableDataset)


def sized_len(dataloader: object) -> Optional[int]:
    """Try to get the length of an object, return ``None`` otherwise."""
    try:
        # try getting the length
        length = len(dataloader)  # type: ignore [arg-type]
    except (TypeError, NotImplementedError):
        length = None
    return length


def has_len(dataloader: object) -> TypeGuard[Sized]:
    """Checks if a given object has ``__len__`` method implemented."""
    length = sized_len(dataloader)
    if length == 0:
        rank_zero_warn(
            f"`{dataloader.__class__.__name__}` returned 0 length. Please make sure this was your intention."
        )
    if length is not None and has_iterable_dataset(dataloader):
        rank_zero_warn(
            "Your `IterableDataset` has `__len__` defined."
            " In combination with multi-process data loading (when num_workers > 1),"
            " `__len__` could be inaccurate if each worker is not configured independently"
            " to avoid having duplicate data."
        )
    return length is not None


def _update_dataloader(dataloader: DataLoader, sampler: Union[Sampler, Iterable]) -> DataLoader:
    dl_args, dl_kwargs = _get_dataloader_init_args_and_kwargs(dataloader, sampler)
    return _reinstantiate_wrapped_cls(dataloader, *dl_args, **dl_kwargs)


def _get_dataloader_init_args_and_kwargs(
    dataloader: DataLoader,
    sampler: Union[Sampler, Iterable],
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
        dl_kwargs.update(_dataloader_init_kwargs_resolve_sampler(dataloader, sampler))

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
            raise TypeError(
                f"Trying to inject parameters into the `{dataloader_cls_name}` instance. "
                "This would fail as it doesn't expose all its attributes in the `__init__` signature. "
                f"The missing arguments are {sorted_missing_kwargs}. HINT: If you wrote the `{dataloader_cls_name}` "
                "class, add the `__init__` arguments or allow passing `**kwargs`"
            )

    return dl_args, dl_kwargs


def _dataloader_init_kwargs_resolve_sampler(
    dataloader: DataLoader,
    sampler: Union[Sampler, Iterable],
) -> dict[str, Any]:
    """This function is used to handle the sampler, batch_sampler arguments associated within a DataLoader for its re-
    instantiation."""
    batch_sampler = getattr(dataloader, "batch_sampler")

    if batch_sampler is not None and type(batch_sampler) is not BatchSampler:
        batch_sampler_cls = type(batch_sampler)
        if hasattr(batch_sampler, "__pl_saved_args"):
            # This is a PyTorch `BatchSampler` subclass for which we captured the init args
            args = batch_sampler.__pl_saved_args
            kwargs = batch_sampler.__pl_saved_kwargs
            default_kwargs = batch_sampler.__pl_saved_default_kwargs
            arg_names = batch_sampler.__pl_saved_arg_names

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
                    drop_last=batch_sampler.drop_last,
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
                    " subclass PyTorch's `BatchSampler`. To mitigate this, either follow the API of `BatchSampler`"
                    " or set`.setup_dataloaders(..., use_distributed_sampler=False)`. If you choose the latter, you"
                    " will be responsible for handling the distributed sampling within your batch sampler."
                ) from ex
        else:
            # The sampler is not a PyTorch `BatchSampler`, we don't know how to inject a custom sampler
            raise TypeError(
                " Lightning can't inject a (distributed) sampler into your batch sampler, because it doesn't"
                " subclass PyTorch's `BatchSampler`. To mitigate this, either follow the API of `BatchSampler`"
                " or set`.setup_dataloaders(..., use_distributed_sampler=False)`. If you choose the latter, you"
                " will be responsible for handling the distributed sampling within your batch sampler."
            )

        return {
            "sampler": None,
            "shuffle": False,
            "batch_sampler": batch_sampler,
            "batch_size": 1,
            "drop_last": False,
        }

    return {"sampler": sampler, "shuffle": False, "batch_sampler": None}


def _auto_add_worker_init_fn(dataloader: object, rank: int) -> None:
    if not hasattr(dataloader, "worker_init_fn"):
        return
    if int(os.environ.get("PL_SEED_WORKERS", 0)) and dataloader.worker_init_fn is None:
        dataloader.worker_init_fn = partial(pl_worker_init_function, rank=rank)


def _reinstantiate_wrapped_cls(orig_object: Any, *args: Any, explicit_cls: Optional[type] = None, **kwargs: Any) -> Any:
    constructor = type(orig_object) if explicit_cls is None else explicit_cls

    try:
        result = constructor(*args, **kwargs)
    except TypeError as ex:
        # improve exception message due to an incorrect implementation of the `DataLoader` where multiple subclass
        # `__init__` arguments map to one `DataLoader.__init__` argument
        import re

        match = re.match(r".*__init__\(\) got multiple values .* '(\w+)'", str(ex))
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
        raise MisconfigurationException(message) from ex

    attrs_record = getattr(orig_object, "__pl_attrs_record", [])
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
    """Wraps the ``__setattr__`` or ``__delattr__`` method of classes (currently :class:`~torch.utils.data.DataLoader`
    and :class:`~torch.utils.data.BatchSampler`) in order to enable re- instantiation of custom subclasses."""

    @functools.wraps(method)
    def wrapper(obj: Any, *args: Any) -> None:
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
            attrs_record = getattr(obj, "__pl_attrs_record", [])
            attrs_record.append((args, tag))
            object.__setattr__(obj, "__pl_attrs_record", attrs_record)
        object.__setattr__(obj, "__pl_current_call", (prev_call_name, prev_call_method))

    return wrapper


@contextmanager
def _replace_dunder_methods(base_cls: type, store_explicit_arg: Optional[str] = None) -> Generator[None, None, None]:
    """This context manager is used to add support for re-instantiation of custom (subclasses) of `base_cls`.

    It patches the ``__init__``, ``__setattr__`` and ``__delattr__`` methods.

    """
    classes = get_all_subclasses(base_cls) | {base_cls}
    for cls in classes:
        # Check that __init__ belongs to the class
        # https://stackoverflow.com/a/5253424
        if "__init__" in cls.__dict__:
            cls.__old__init__ = cls.__init__  # type: ignore[misc]
            cls.__init__ = _wrap_init_method(cls.__init__, store_explicit_arg)  # type: ignore[misc]

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


def _replace_value_in_saved_args(
    replace_key: str,
    replace_value: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    default_kwargs: dict[str, Any],
    arg_names: tuple[str, ...],
) -> tuple[bool, tuple[Any, ...], dict[str, Any]]:
    """Tries to replace an argument value in a saved list of args and kwargs.

    Returns a tuple indicating success of the operation and modified saved args and kwargs

    """

    if replace_key in arg_names:
        replace_index = arg_names.index(replace_key)
        args = args[:replace_index] + (replace_value,) + args[replace_index + 1 :]
        return True, args, kwargs
    if replace_key in kwargs or replace_key in default_kwargs:
        kwargs[replace_key] = replace_value
        return True, args, kwargs

    return False, args, kwargs


def _set_sampler_epoch(dataloader: object, epoch: int) -> None:
    """Calls the ``set_epoch`` method on either the sampler of the given dataloader.

    Every PyTorch dataloader has either a sampler or a batch sampler. If the sampler is wrapped by a
    :class:`~torch.utils.data.distributed.DistributedSampler`, ``set_epoch`` must be called at the beginning
    of every epoch to ensure shuffling applies a new ordering. This has no effect if shuffling is off.

    """
    # cannot use a set because samplers might be unhashable: use a dict based on the id to drop duplicates
    objects: dict[int, Any] = {}
    # check dataloader.sampler
    if (sampler := getattr(dataloader, "sampler", None)) is not None:
        objects[id(sampler)] = sampler
    # check dataloader.batch_sampler.sampler
    if (batch_sampler := getattr(dataloader, "batch_sampler", None)) is not None and (
        sampler := getattr(batch_sampler, "sampler", None)
    ) is not None:
        objects[id(sampler)] = sampler
    for obj in objects.values():
        set_epoch = getattr(obj, "set_epoch", None)
        if callable(set_epoch):
            set_epoch(epoch)


def suggested_max_num_workers(local_world_size: int) -> int:
    """Suggests an upper bound of ``num_workers`` to use in a PyTorch :class:`~torch.utils.data.DataLoader` based on
    the number of CPU cores available on the system and the number of distributed processes in the current machine.

    Args:
        local_world_size: The number of distributed processes running on the current machine. Set this to the number
            of devices configured in Fabric/Trainer.

    """
    if local_world_size < 1:
        raise ValueError(f"`local_world_size` should be >= 1, got {local_world_size}.")
    cpu_count = _num_cpus_available()
    return max(1, cpu_count // local_world_size - 1)  # -1 to leave some resources for main process


def _num_cpus_available() -> int:
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))

    cpu_count = os.cpu_count()
    return 1 if cpu_count is None else cpu_count


class AttributeDict(dict):
    """A container to store state variables of your program.

    This is a drop-in replacement for a Python dictionary, with the additional functionality to access and modify keys
    through attribute lookup for convenience.

    Use this to define the state of your program, then pass it to
    :meth:`~lightning.fabric.fabric.Fabric.save` and :meth:`~lightning.fabric.fabric.Fabric.load`.

    Example:
        >>> import torch
        >>> model = torch.nn.Linear(2, 2)
        >>> state = AttributeDict(model=model, iter_num=0)
        >>> state.model
        Linear(in_features=2, out_features=2, bias=True)
        >>> state.iter_num += 1
        >>> state.iter_num
        1
        >>> state
        "iter_num": 1
        "model":    Linear(in_features=2, out_features=2, bias=True)

    """

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'") from e

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __delattr__(self, item: str) -> None:
        if item not in self:
            raise KeyError(item)
        del self[item]

    def __repr__(self) -> str:
        if not len(self):
            return ""
        max_key_length = max(len(str(k)) for k in self)
        tmp_name = "{:" + str(max_key_length + 3) + "s} {}"
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        return "\n".join(rows)
