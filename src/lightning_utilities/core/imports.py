# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import functools
import importlib
import operator
import os
import warnings
from functools import lru_cache
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, List, Optional

import pkg_resources
from packaging.requirements import Requirement
from packaging.version import Version

try:
    from importlib import metadata
except ImportError:
    # Python < 3.8
    import importlib_metadata as metadata  # type: ignore


@lru_cache()
def package_available(package_name: str) -> bool:
    """Check if a package is available in your environment.

    >>> package_available('os')
    True
    >>> package_available('bla')
    False
    """
    try:
        return find_spec(package_name) is not None
    except ModuleNotFoundError:
        return False


@lru_cache()
def module_available(module_path: str) -> bool:
    """Check if a module path is available in your environment.

    >>> module_available('os')
    True
    >>> module_available('os.bla')
    False
    >>> module_available('bla.bla')
    False
    """
    module_names = module_path.split(".")
    if not package_available(module_names[0]):
        return False
    try:
        importlib.import_module(module_path)
    except ImportError:
        return False
    return True


def compare_version(package: str, op: Callable, version: str, use_base_version: bool = False) -> bool:
    """Compare package version with some requirements.

    >>> compare_version("torch", operator.ge, "0.1")
    True
    >>> compare_version("does_not_exist", operator.ge, "0.0")
    False
    """
    try:
        pkg = importlib.import_module(package)
    except (ImportError, pkg_resources.DistributionNotFound):
        return False
    try:
        if hasattr(pkg, "__version__"):
            pkg_version = Version(pkg.__version__)
        else:
            # try pkg_resources to infer version
            pkg_version = Version(pkg_resources.get_distribution(package).version)
    except TypeError:
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))


class RequirementCache:
    """Boolean-like class for check of requirement with extras and version specifiers.

    >>> RequirementCache("torch>=0.1")
    Requirement 'torch>=0.1' met
    >>> bool(RequirementCache("torch>=0.1"))
    True
    >>> bool(RequirementCache("torch>100.0"))
    False
    """

    def __init__(self, requirement: str) -> None:
        self.requirement = requirement

    def _check_requirement(self) -> None:
        if not hasattr(self, "available"):
            try:
                pkg_resources.require(self.requirement)
                self.available = True
                self.message = f"Requirement {self.requirement!r} met"
            except Exception as ex:
                self.available = False
                self.message = f"{ex.__class__.__name__}: {ex}. HINT: Try running `pip install -U {self.requirement!r}`"

    def __bool__(self) -> bool:
        self._check_requirement()
        return self.available

    def __str__(self) -> str:
        self._check_requirement()
        return self.message

    def __repr__(self) -> str:
        return self.__str__()


def get_dependency_min_version_spec(package_name: str, dependency_name: str) -> str:
    """Returns the minimum version specifier of a dependency of a package.

    >>> get_dependency_min_version_spec("pytorch-lightning==1.8.0", "jsonargparse")
    '>=4.12.0'
    """
    dependencies = metadata.requires(package_name) or []
    for dep in dependencies:
        dependency = Requirement(dep)
        if dependency.name == dependency_name:
            spec = [str(s) for s in dependency.specifier if str(s)[0] == ">"]
            return spec[0] if spec else ""
    raise ValueError(
        "This is an internal error. Please file a GitHub issue with the error message. Dependency "
        f"{dependency_name!r} not found in package {package_name!r}."
    )


class LazyModule(ModuleType):
    """Proxy module that lazily imports the underlying module the first time it is actually used.

    Args:
        module_name: the fully-qualified module name to import
        callback: a callback function to call before importing the module
    """

    def __init__(self, module_name: str, callback: Optional[Callable] = None) -> None:
        super().__init__(module_name)
        self._module: Any = None
        self._callback = callback

    def __getattr__(self, item: str) -> Any:
        if self._module is None:
            self._import_module()

        return getattr(self._module, item)

    def __dir__(self) -> List[str]:
        if self._module is None:
            self._import_module()

        return dir(self._module)

    def _import_module(self) -> None:
        # Execute callback, if any
        if self._callback is not None:
            self._callback()

        # Actually import the module
        self._module = importlib.import_module(self.__name__)

        # Update this object's dict so that attribute references are efficient
        # (__getattr__ is only called on lookups that fail)
        self.__dict__.update(self._module.__dict__)


def lazy_import(module_name: str, callback: Optional[Callable] = None) -> LazyModule:
    """Returns a proxy module object that will lazily import the given module the first time it is used. Example usage::

        # Lazy version of `import tensorflow as tf`
        tf = lazy_import("tensorflow")
        # Other commands
        # Now the module is loaded
        tf.__version__
    Args:
        module_name: the fully-qualified module name to import
        callback: a callback function to call before importing the module
    Returns:
        a proxy module object that will be lazily imported when first used
    """
    return LazyModule(module_name, callback=callback)


def requires(*module_path: str, raise_exception: bool = True) -> Callable:
    """Wrapper for early import failure with some nice exception message.

    Example:

        >>> @requires("libpath", raise_exception=bool(int(os.getenv("LIGHTING_TESTING", "0"))))
        ... def my_cwd():
        ...     from pathlib import Path
        ...     return Path(__file__).parent

        >>> class MyRndPower:
        ...     @requires("math", "random")
        ...     def __init__(self):
        ...         from math import pow
        ...         from random import randint
        ...         self._rnd = pow(randint(1, 9), 2)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            unavailable_modules = [module for module in module_path if not module_available(module)]
            if any(unavailable_modules):
                msg = f"Required dependencies not available. Please run `pip install {' '.join(unavailable_modules)}`"
                if raise_exception:
                    raise ModuleNotFoundError(msg)
                warnings.warn(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator
