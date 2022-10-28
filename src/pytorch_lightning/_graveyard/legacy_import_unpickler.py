import pickle
import warnings
from typing import Any, Callable

import torchmetrics
from lightning_utilities.core.imports import compare_version as _compare_version


def _patch_pl_to_mirror_if_necessary(module: str) -> str:
    pl = "pytorch_" + "lightning"  # avoids replacement during mirror package generation
    if module.startswith(pl):
        # for the standalone package this won't do anything,
        # for the unified mirror package it will redirect the imports
        module = "pytorch_lightning" + module[len(pl) :]
    return module


class RedirectingUnpickler(pickle._Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        new_module = _patch_pl_to_mirror_if_necessary(module)
        # this warning won't trigger for standalone as these imports are identical
        if module != new_module:
            warnings.warn(f"Redirecting import of {module}.{name} to {new_module}.{name}")
        return super().find_class(new_module, name)


def compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    new_package = _patch_pl_to_mirror_if_necessary(package)
    return _compare_version(new_package, op, version, use_base_version)


try:
    if hasattr(torchmetrics.utilities.imports, "_compare_version"):
        torchmetrics.utilities.imports._compare_version = compare_version  # type: ignore
except AttributeError:
    pass

try:
    if hasattr(torchmetrics.metric, "_compare_version"):
        torchmetrics.metric._compare_version = compare_version  # type: ignore
except AttributeError:
    pass
pickle.Unpickler = RedirectingUnpickler  # type: ignore
