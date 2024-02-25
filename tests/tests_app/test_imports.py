import importlib
import inspect
import os
import types
from typing import TypeVar

import lightning.app


def _is_attribute(member, module):
    return all([
        hasattr(member, "__module__") and module.__name__ in member.__module__,
        not isinstance(member, TypeVar),
        not isinstance(member, types.ModuleType),
    ])


def _find_exports(module):
    members = inspect.getmembers(module)
    attributes = {member[0] for member in members if _is_attribute(member[1], module)}
    public_attributes = list(filter(lambda attribute: not attribute.startswith("_"), attributes))
    exports = {attribute: module.__name__ for attribute in public_attributes}

    if module.__file__ is not None and "__init__.py" in module.__file__:
        root = os.path.dirname(module.__file__)
        submodule_paths = os.listdir(root)
        submodule_paths = [path for path in submodule_paths if not path.startswith("_")]
        submodules = [
            path.replace(".py", "")
            for path in submodule_paths
            if os.path.isdir(os.path.join(root, path)) or path.endswith(".py")
        ]
        for submodule in submodules:
            deeper_exports = _find_exports(importlib.import_module(f".{submodule}", module.__name__))
            exports = {**deeper_exports, **exports}

    return exports or {}


def test_import_depth(
    ignore=[
        "lightning.app.cli",
        "lightning.app.components.serve.types",
        "lightning.app.core",
        "lightning.app.launcher",
        "lightning.app.runners",
        "lightning.app.utilities",
    ],
):
    """This test ensures that any public exports (functions, classes, etc.) can be imported by users with at most a
    depth of two. This guarantees that everything user-facing can be imported with (at most) ``lightning.app.*.*``.

    Args:
        ignore: Sub-module paths to ignore (usually sub-modules that are not intended to be user-facing).

    """
    exports = _find_exports(lightning.app)
    depths = {export: len(path.replace("lightning.app", "").split(".")) for export, path in exports.items()}
    deep_exports = [export for export, depth in depths.items() if depth > 2]
    deep_exports = list(
        filter(lambda export: not any(exports[export].startswith(path) for path in ignore), deep_exports)
    )
    if len(deep_exports) > 0:
        raise RuntimeError(
            "Found exports with a depth greater than two. "
            "Either expose them at a higher level or make them private. "
            f"Found: {', '.join(sorted(f'{exports[export]}.{export}' for export in deep_exports))}"
        )
