import pickle
import warnings
from copy import deepcopy
from typing import Any


class RedirectingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("pytorch_" + "lightning"):
            # for the standalone package this won't do anything,
            # for the unified mirror package it will redirect the imports

            old_module = deepcopy(module)
            module = "pytorch_lightning" + module[len("pytorch_" + "lightning") :]

            # this warning won't trigger for standalone as these imports are identical
            if module != old_module:
                warnings.warn(f"Redirecting import of {module}.{name} to pytorch_lightning.{name}")

        return super().find_class(module, name)


pickle.Unpickler = RedirectingUnpickler  # type: ignore
