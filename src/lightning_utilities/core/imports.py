# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0

import functools
import importlib
import os
import warnings
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, distribution
from importlib.metadata import version as _version
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Callable, Optional, TypeVar

from packaging.requirements import Requirement
from packaging.version import InvalidVersion, Version
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")

try:
    from importlib import metadata
except ImportError:
    # Python < 3.8
    import importlib_metadata as metadata  # type: ignore


@lru_cache
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


@lru_cache
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
    except ImportError:
        return False
    try:
        # Use importlib.metadata to infer version
        pkg_version = Version(pkg.__version__) if hasattr(pkg, "__version__") else Version(_version(package))
    except (TypeError, PackageNotFoundError):
        # this is mocked by Sphinx, so it should return True to generate all summaries
        return True
    if use_base_version:
        pkg_version = Version(pkg_version.base_version)
    return op(pkg_version, Version(version))


class RequirementCache:
    """Boolean-like class to check for requirement and module availability.

    Args:
        requirement: The requirement to check, version specifiers are allowed.
        module: The optional module to try to import if the requirement check fails.

    >>> RequirementCache("torch>=0.1")
    Requirement 'torch>=0.1' met
    >>> bool(RequirementCache("torch>=0.1"))
    True
    >>> bool(RequirementCache("torch>100.0"))
    False
    >>> RequirementCache("torch")
    Requirement 'torch' met
    >>> bool(RequirementCache("torch"))
    True
    >>> bool(RequirementCache("unknown_package"))
    False
    >>> bool(RequirementCache(module="torch.utils"))
    True
    >>> bool(RequirementCache(module="unknown_package"))
    False
    >>> bool(RequirementCache(module="unknown.module.path"))
    False

    """

    def __init__(self, requirement: Optional[str] = None, module: Optional[str] = None) -> None:
        if not (requirement or module):
            raise ValueError("At least one arguments need to be set.")
        self.requirement = requirement
        self.module = module

    def _check_requirement(self) -> None:
        assert self.requirement  # noqa: S101; needed for typing
        try:
            req = Requirement(self.requirement)
            pkg_version = Version(_version(req.name))
            self.available = req.specifier.contains(pkg_version, prereleases=True) and (
                not req.extras or self._check_extras_available(req)
            )
        except (PackageNotFoundError, InvalidVersion) as ex:
            self.available = False
            self.message = f"{ex.__class__.__name__}: {ex}. HINT: Try running `pip install -U {self.requirement!r}`"

        if self.available:
            self.message = f"Requirement {self.requirement!r} met"
        else:
            req_include_version = any(c in self.requirement for c in "=<>")
            if not req_include_version or self.module is not None:
                module = self.requirement if self.module is None else self.module
                # Sometimes `importlib.metadata.version` fails but the module is importable
                self.available = module_available(module)
                if self.available:
                    self.message = f"Module {module!r} available"
            self.message = (
                f"Requirement {self.requirement!r} not met. HINT: Try running `pip install -U {self.requirement!r}`"
            )

    def _check_module(self) -> None:
        assert self.module  # noqa: S101; needed for typing
        self.available = module_available(self.module)
        if self.available:
            self.message = f"Module {self.module!r} available"
        else:
            self.message = f"Module not found: {self.module!r}. HINT: Try running `pip install -U {self.module}`"

    def _check_available(self) -> None:
        if hasattr(self, "available"):
            return
        if self.requirement:
            self._check_requirement()
        if getattr(self, "available", True) and self.module:
            self._check_module()

    def _check_extras_available(self, requirement: Requirement) -> bool:
        if not requirement.extras:
            return True

        extra_requirements = self._get_extra_requirements(requirement)

        if not extra_requirements:
            # The specified extra is not found in the package metadata
            return False

        # Verify each extra requirement is installed
        for extra_req in extra_requirements:
            try:
                extra_dist = distribution(extra_req.name)
                extra_installed_version = Version(extra_dist.version)
                if extra_req.specifier and not extra_req.specifier.contains(extra_installed_version, prereleases=True):
                    return False
            except importlib.metadata.PackageNotFoundError:
                return False

        return True

    def _get_extra_requirements(self, requirement: Requirement) -> list[Requirement]:
        dist = distribution(requirement.name)
        # Get the required dependencies for the specified extras
        extra_requirements = dist.metadata.get_all("Requires-Dist") or []
        return [Requirement(r) for r in extra_requirements if any(extra in r for extra in requirement.extras)]

    def __bool__(self) -> bool:
        """Format as bool."""
        self._check_available()
        return self.available

    def __str__(self) -> str:
        """Format as string."""
        self._check_available()
        return self.message

    def __repr__(self) -> str:
        """Format as string."""
        return self.__str__()


class ModuleAvailableCache(RequirementCache):
    """Boolean-like class for check of module availability.

    >>> ModuleAvailableCache("torch")
    Module 'torch' available
    >>> bool(ModuleAvailableCache("torch.utils"))
    True
    >>> bool(ModuleAvailableCache("unknown_package"))
    False
    >>> bool(ModuleAvailableCache("unknown.module.path"))
    False

    """

    def __init__(self, module: str) -> None:
        warnings.warn(
            "`ModuleAvailableCache` is a special case of `RequirementCache`."
            " Please use `RequirementCache(module=...)` instead.",
            DeprecationWarning,
            stacklevel=4,
        )
        super().__init__(module=module)


def get_dependency_min_version_spec(package_name: str, dependency_name: str) -> str:
    """Return the minimum version specifier of a dependency of a package.

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
        """Overwrite attribute access to attribute."""
        if self._module is None:
            self._import_module()

        return getattr(self._module, item)

    def __dir__(self) -> list[str]:
        """Overwrite attribute access for dictionary."""
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
    """Return a proxy module object that will lazily import the given module the first time it is used.

    Example usage:

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


def requires(*module_path_version: str, raise_exception: bool = True) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Wrap early import failure with some nice exception message.

    Args:
        module_path_version: python package path (e.g. `torch.cuda`) or pip like requiremsnt (e.g. `torch>=2.0.0`)
        raise_exception: how strict the check shall be if exit the code or just warn user

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

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        reqs = [
            ModuleAvailableCache(mod_ver) if "." in mod_ver else RequirementCache(mod_ver)
            for mod_ver in module_path_version
        ]
        available = all(map(bool, reqs))

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not available:
                missing = os.linesep.join([repr(r) for r in reqs if not bool(r)])
                msg = f"Required dependencies not available: \n{missing}"
                if raise_exception:
                    raise ModuleNotFoundError(msg)
                warnings.warn(msg, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator
