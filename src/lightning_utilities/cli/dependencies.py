# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import glob
import os.path
import re
import warnings
from collections.abc import Sequence
from pprint import pprint
from typing import Union

REQUIREMENT_ROOT = "requirements.txt"
REQUIREMENT_FILES_ALL: list = glob.glob(os.path.join("requirements", "*.txt"))
REQUIREMENT_FILES_ALL += glob.glob(os.path.join("requirements", "**", "*.txt"), recursive=True)
REQUIREMENT_FILES_ALL += glob.glob(os.path.join("**", "pyproject.toml"))
if os.path.isfile(REQUIREMENT_ROOT):
    REQUIREMENT_FILES_ALL += [REQUIREMENT_ROOT]


def prune_packages_in_requirements(
    packages: Union[str, Sequence[str]], req_files: Union[str, Sequence[str]] = REQUIREMENT_FILES_ALL
) -> None:
    """Remove some packages from given requirement files."""
    if isinstance(packages, str):
        packages = [packages]
    if isinstance(req_files, str):
        req_files = [req_files]
    for req in req_files:
        _prune_packages(req, packages)


def _prune_packages(req_file: str, packages: Sequence[str]) -> None:
    """Remove some packages from given requirement files."""
    with open(req_file) as fp:
        lines = fp.readlines()

    if isinstance(packages, str):
        packages = [packages]
    for pkg in packages:
        lines = [ln for ln in lines if not ln.startswith(pkg)]
    pprint(lines)

    with open(req_file, "w") as fp:
        fp.writelines(lines)


def _replace_min_req_in_txt(req_file: str) -> None:
    with open(req_file) as fopen:
        req = fopen.read().replace(">=", "==")
    with open(req_file, "w") as fw:
        fw.write(req)


def _replace_min_req_in_pyproject_toml(proj_file: str = "pyproject.toml") -> None:
    """Replace all `>=` with `==` in the standard pyproject.toml file in [project.dependencies]."""
    import tomlkit

    # Load and parse the existing pyproject.toml
    with open(proj_file, encoding="utf-8") as f:
        content = f.read()
    doc = tomlkit.parse(content)

    # todo: consider also replace extras in [dependency-groups] -> extras = [...]
    deps = doc.get("project", {}).get("dependencies")
    if not deps:
        return

    # Replace '>=version' with '==version' in each dependency
    for i, req in enumerate(deps):
        # Simple string value
        deps[i] = req.replace(">=", "==")

    # Dump back out, preserving layout
    with open(proj_file, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))


def replace_oldest_version(req_files: Union[str, Sequence[str]] = REQUIREMENT_FILES_ALL) -> None:
    """Replace the min package version by fixed one."""
    if isinstance(req_files, str):
        req_files = [req_files]
    for fname in req_files:
        if fname.endswith(".txt"):
            _replace_min_req_in_txt(fname)
        elif os.path.basename(fname) == "pyproject.toml":
            _replace_min_req_in_pyproject_toml(fname)
        else:
            warnings.warn(
                "Only *.txt with plain list of requirements or standard pyproject.toml are supported."
                f"Provided '{fname}' is not supported.",
                UserWarning,
                stacklevel=2,
            )


def _replace_package_name_in_txt(req_file: str, old_package: str, new_package: str) -> None:
    """Replace one package by another with the same version in a given requirement file."""
    # load file
    with open(req_file) as fopen:
        requirements = fopen.readlines()
    # replace all occurrences
    for i, req in enumerate(requirements):
        requirements[i] = re.sub(r"^" + re.escape(old_package) + r"(?=[ <=>#]|$)", new_package, req)
    # save file
    with open(req_file, "w") as fw:
        fw.writelines(requirements)


def _replace_package_name_in_pyproject_toml(proj_file: str, old_package: str, new_package: str) -> None:
    """Replace one package by another with the same version in the standard pyproject.toml file."""
    import tomlkit

    # Load and parse the existing pyproject.toml
    with open(proj_file, encoding="utf-8") as f:
        content = f.read()
    doc = tomlkit.parse(content)

    # todo: consider also replace extras in [dependency-groups] -> extras = [...]
    deps = doc.get("project", {}).get("dependencies")
    if not deps:
        return

    # Replace '>=version' with '==version' in each dependency
    for i, req in enumerate(deps):
        # Simple string value
        deps[i] = re.sub(r"^" + re.escape(old_package) + r"(?=[ <=>]|$)", new_package, req)

    # Dump back out, preserving layout
    with open(proj_file, "w", encoding="utf-8") as f:
        f.write(tomlkit.dumps(doc))


def replace_package_in_requirements(
    old_package: str, new_package: str, req_files: Union[str, Sequence[str]] = REQUIREMENT_FILES_ALL
) -> None:
    """Replace one package by another with same version in given requirement files."""
    if isinstance(req_files, str):
        req_files = [req_files]
    for fname in req_files:
        if fname.endswith(".txt"):
            _replace_package_name_in_txt(fname, old_package, new_package)
        elif os.path.basename(fname) == "pyproject.toml":
            _replace_package_name_in_pyproject_toml(fname, old_package, new_package)
        else:
            warnings.warn(
                "Only *.txt with plain list of requirements or standard pyproject.toml are supported."
                f"Provided '{fname}' is not supported.",
                UserWarning,
                stacklevel=2,
            )
