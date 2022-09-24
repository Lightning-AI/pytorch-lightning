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
from contextlib import contextmanager
from typing import Any, Callable, Generator, Mapping, Optional, Set, Type, Union

from lightning_utilities.core.imports import module_available
from torch import Tensor
from torch.nn import Module, Parameter

from pytorch_lightning.utilities.rank_zero import rank_zero_deprecation


def is_meta_init() -> bool:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.meta.is_meta_init` is deprecated in v1.8 and will be removed in v1.9."
        " The function has become a no-op."
        " Please check out the `torchdistx` project instead: https://github.com/pytorch/torchdistx"
    )
    return False


def init_meta(module_fn: Callable[..., Module], *args: Any, **kwargs: Any) -> None:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.meta.init_meta` is deprecated in v1.8 and will be removed in v1.9."
        " The function has become a no-op."
        " Please check out the `torchdistx` project instead: https://github.com/pytorch/torchdistx"
    )


def get_all_subclasses(cls: Type) -> Set[Type]:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.meta.get_all_subclasses` is deprecated in v1.8 and will be removed in v1.9."
        " Please copy its implementation if you have a use for it."
    )
    from lightning_utilities.core.inheritance import get_all_subclasses as new_get_all_subclasses

    return new_get_all_subclasses(cls)


def recursively_setattr(root_module: Any, prefix: str, materialized_module: Module) -> None:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.meta.recursively_setattr` is deprecated in v1.8 and will be removed in v1.9."
        " Please copy its implementation if you have a use for it."
    )
    *path, name = prefix.split(".")
    for p in path:
        root_module = getattr(root_module, p)

    try:
        index = int(name)
        root_module[index] = materialized_module
    except ValueError:
        setattr(root_module, name, materialized_module)


def materialize_module(root_module: Module) -> None:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.meta.materialize_module` is deprecated in v1.8 and will be removed in v1.9."
        " The function has become a no-op."
        " Please check out the `torchdistx` project instead: https://github.com/pytorch/torchdistx"
    )


@contextmanager
def init_meta_context() -> Generator:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.meta.init_meta_context` is deprecated in v1.8 and will be removed in v1.9."
        " The function has become a no-op."
        " Please check out the `torchdistx` project instead: https://github.com/pytorch/torchdistx"
    )
    yield


def is_on_meta_device(module: Module) -> bool:
    rank_zero_deprecation(
        "`pytorch_lightning.utilities.meta.is_on_meta_device` is deprecated in v1.8 and will be removed in v1.9."
        " Please copy its implementation if you have a use for it."
    )
    try:
        param = next(module.parameters())
        return param.is_meta
    except StopIteration:
        return False


def _is_deferred(module: Optional[Module]) -> bool:
    if module is None or not module_available("torchdistx.fake"):
        return False
    from torchdistx.fake import is_fake

    def any_fake(tensors: Mapping[str, Optional[Union[Tensor, Parameter]]]) -> bool:
        return any(is_fake(t) for t in tensors.values() if t is not None)

    is_deferred = any(_is_deferred(m) for m in module.children())
    return is_deferred or any_fake(module._parameters) or any_fake(module._buffers)
