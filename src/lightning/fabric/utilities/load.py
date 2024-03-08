# Copyright 2023 MathInf GmbH
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this files from this repository except in compliance
# with the License reproduced below (also at
# http://www.apache.org/licenses/LICENSE-2.0).
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import pickle
import warnings
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Optional, OrderedDict, Sequence, Set, Union

import torch
from lightning_utilities.core.apply_func import apply_to_collection
from torch import Tensor
from torch._C import _TensorMeta
from torch.nn import Parameter
from typing_extensions import override

from lightning.fabric.utilities.imports import (
    _TORCH_GREATER_EQUAL_2_0,
    _TORCH_GREATER_EQUAL_2_3,
)
from lightning.fabric.utilities.types import _PATH, _Stateful

_METADATA_FILENAME = "meta.pt"


if TYPE_CHECKING:
    from torch.storage import TypedStorage


# Modified from https://github.com/lernapparat/torchhacks by Thomas Viehmann
class _NotYetLoadedTensor:
    def __init__(
        self,
        metatensor: Tensor,
        archiveinfo: "_LazyLoadingUnpickler",
        storageinfo: tuple,
        rebuild_args: tuple,
    ) -> None:
        self.metatensor = metatensor
        self.archiveinfo = archiveinfo
        self.storageinfo = storageinfo
        self.rebuild_args = rebuild_args

    @classmethod
    def rebuild_from_type_v2(
        cls,
        func: Callable,
        new_type: _TensorMeta,
        args: tuple,
        state: dict,
        *,
        archiveinfo: Optional["_LazyLoadingUnpickler"] = None,
    ) -> Any:
        ret = func(*args)
        if isinstance(ret, _NotYetLoadedTensor):
            old_lt = ret._load_tensor

            def _load_tensor() -> Any:
                t = old_lt()
                return torch._tensor._rebuild_from_type_v2(lambda: t, new_type, (), state)

            ret._load_tensor = _load_tensor  # type: ignore[method-assign]
            return ret
        return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)

    @classmethod
    def rebuild_parameter(
        cls,
        data: Any,
        requires_grad: bool,
        backward_hooks: OrderedDict,
        *,
        archiveinfo: Optional["_LazyLoadingUnpickler"] = None,
    ) -> Union[Tensor, "_NotYetLoadedTensor"]:
        if isinstance(data, _NotYetLoadedTensor):
            old_lt = data._load_tensor

            def _load_tensor() -> Parameter:
                t = old_lt()
                return torch._utils._rebuild_parameter(t, requires_grad, backward_hooks)

            data._load_tensor = _load_tensor  # type: ignore[method-assign]
            return data
        return torch._utils._rebuild_parameter(data, requires_grad, backward_hooks)

    @classmethod
    def rebuild_tensor_v2(
        cls,
        storage: "TypedStorage",
        storage_offset: int,
        size: tuple,
        stride: tuple,
        requires_grad: bool,
        backward_hooks: OrderedDict,
        metadata: Optional[Any] = None,
        *,
        archiveinfo: "_LazyLoadingUnpickler",
    ) -> "_NotYetLoadedTensor":
        rebuild_args = (storage_offset, size, stride, requires_grad, backward_hooks, metadata)
        metatensor = torch._utils._rebuild_tensor_v2(
            storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata
        )
        storageinfo = storage.archiveinfo
        return _NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)

    def _load_tensor(self) -> Tensor:
        from torch.storage import TypedStorage, UntypedStorage

        _, _, fn, _, size = self.storageinfo
        dtype = self.metatensor.dtype

        storage = self.archiveinfo.file_reader.get_storage_from_record(
            f"data/{fn}", size * torch._utils._element_size(dtype), UntypedStorage
        )
        uts = storage._typed_storage()._untyped_storage

        with warnings.catch_warnings():
            # The TypedStorage APIs have heavy deprecations in torch, suppress all these warnings for now
            warnings.simplefilter("ignore")
            storage = TypedStorage(wrap_storage=uts, dtype=dtype, _internal=True)
        return torch._utils._rebuild_tensor_v2(storage, *self.rebuild_args)

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: Sequence,
        args: Sequence[Any] = (),
        kwargs: Optional[Dict] = None,
    ) -> Any:
        kwargs = kwargs or {}
        loaded_args = [(arg._load_tensor() if isinstance(arg, _NotYetLoadedTensor) else arg) for arg in args]
        return func(*loaded_args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # These properties don't require materialization and can be accessed through the meta tensor directly
        if name in {
            "dtype",
            "grad",
            "grad_fn",
            "is_meta",
            "layout",
            "names",
            "ndim",
            "output_nr",
            "requires_grad",
            "retains_grad",
            "size",
            "shape",
            "volatile",
        }:
            return getattr(self.metatensor, name)

        # materializing these is needed for quantization (see lit-gpt)
        if name in {"contiguous", "cuda", "half"}:
            return getattr(self._load_tensor(), name)

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({repr(self.metatensor)})"


# Modified from https://github.com/lernapparat/torchhacks by Thomas Viehmann
class _LazyLoadingUnpickler(pickle.Unpickler):
    def __init__(self, file: IO, file_reader: torch.PyTorchFileReader) -> None:
        super().__init__(file)
        self.file_reader = file_reader

    @override
    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return partial(_NotYetLoadedTensor.rebuild_tensor_v2, archiveinfo=self)
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return partial(_NotYetLoadedTensor.rebuild_from_type_v2, archiveinfo=self)
        if module == "torch._utils" and name == "_rebuild_parameter":
            return partial(_NotYetLoadedTensor.rebuild_parameter, archiveinfo=self)
        return super().find_class(module, name)

    @override
    def persistent_load(self, pid: tuple) -> "TypedStorage":
        from torch.storage import TypedStorage

        _, cls, _, _, _ = pid
        with warnings.catch_warnings():
            # The TypedStorage APIs have heavy deprecations in torch, suppress all these warnings for now
            warnings.simplefilter("ignore")
            storage = TypedStorage(dtype=cls().dtype, device="meta")
        storage.archiveinfo = pid
        return storage


def _lazy_load(filename: _PATH) -> Any:
    if not _TORCH_GREATER_EQUAL_2_0:
        raise NotImplementedError("Lazy-loading is only supported with PyTorch >= 2.0.")
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Path {str(filename)!r} does not exist or is not a file.")
    file_reader = torch.PyTorchFileReader(str(filename))
    with BytesIO(file_reader.get_record("data.pkl")) as pkl:
        mup = _LazyLoadingUnpickler(pkl, file_reader)
        return mup.load()


def _materialize_tensors(collection: Any) -> Any:
    def _load_tensor(t: _NotYetLoadedTensor) -> Tensor:
        return t._load_tensor()

    return apply_to_collection(collection, dtype=_NotYetLoadedTensor, function=_load_tensor)


def _move_state_into(
    source: Dict[str, Any], destination: Dict[str, Union[Any, _Stateful]], keys: Optional[Set[str]] = None
) -> None:
    """Takes the state from the source destination and moves it into the destination dictionary.

    If an object in the destination follows the stateful protocol, it loads the source state via ``load_state_dict``.

    """
    keys = set(source) if keys is None else keys & set(source)
    for key in keys:
        state = source.pop(key)
        if key in destination and isinstance(destination[key], _Stateful):
            destination[key].load_state_dict(state)
        else:
            destination[key] = state


def _load_distributed_checkpoint(checkpoint_folder: Path) -> Dict[str, Any]:
    """Loads a sharded checkpoint saved with the `torch.distributed.checkpoint` into a full state dict.

    The current implementation assumes that the entire checkpoint fits in CPU memory.

    """
    if not _TORCH_GREATER_EQUAL_2_3:
        raise ImportError("Processing distributed checkpoints requires PyTorch >= 2.3.")

    from torch.distributed.checkpoint import FileSystemReader
    from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    checkpoint: Dict[str, Any] = {}
    _load_state_dict(
        checkpoint,
        storage_reader=FileSystemReader(checkpoint_folder),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    # This is the extra file saved by Fabric, with user data separate from weights and optimizer states
    extra_file = checkpoint_folder / _METADATA_FILENAME
    extra = torch.load(extra_file, map_location="cpu") if extra_file.is_file() else {}
    checkpoint.update(extra)

    return checkpoint
