import pickle
import warnings
from typing import Any


class RedirectingUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        pl = "pytorch_" + "lightning"  # avoids replacement during mirror package generation
        if module.startswith(pl):
            # for the standalone package this won't do anything,
            # for the unified mirror package it will redirect the imports
            old_module = module
            module = "pytorch_lightning" + module[len(pl) :]
            # this warning won't trigger for standalone as these imports are identical
            if module != old_module:
                warnings.warn(f"Redirecting import of {old_module}.{name} to {module}.{name}")
        return super().find_class(module, name)


pickle.Unpickler = RedirectingUnpickler  # type: ignore
