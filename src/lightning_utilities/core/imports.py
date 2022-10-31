# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
import importlib
import operator
from functools import lru_cache
from importlib.util import find_spec
from typing import Callable

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

    >>> get_dependency_min_version_spec("pytorch-lightning", "jsonargparse")
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
