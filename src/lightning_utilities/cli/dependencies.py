# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import glob
import os.path
import re
from collections.abc import Sequence
from pprint import pprint
from typing import Union

REQUIREMENT_ROOT = "requirements.txt"
REQUIREMENT_FILES_ALL: list = glob.glob(os.path.join("requirements", "*.txt"))
REQUIREMENT_FILES_ALL += glob.glob(os.path.join("requirements", "**", "*.txt"), recursive=True)
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


def _replace_min(fname: str) -> None:
    with open(fname) as fopen:
        req = fopen.read().replace(">=", "==")
    with open(fname, "w") as fw:
        fw.write(req)


def replace_oldest_version(req_files: Union[str, Sequence[str]] = REQUIREMENT_FILES_ALL) -> None:
    """Replace the min package version by fixed one."""
    if isinstance(req_files, str):
        req_files = [req_files]
    for fname in req_files:
        _replace_min(fname)


def _replace_package_name(requirements: list[str], old_package: str, new_package: str) -> list[str]:
    """Replace one package by another with same version in given requirement file.

    >>> _replace_package_name(["torch>=1.0 # comment", "torchvision>=0.2", "torchtext <0.3"], "torch", "pytorch")
    ['pytorch>=1.0 # comment', 'torchvision>=0.2', 'torchtext <0.3']

    """
    for i, req in enumerate(requirements):
        requirements[i] = re.sub(r"^" + re.escape(old_package) + r"(?=[ <=>#]|$)", new_package, req)
    return requirements


def replace_package_in_requirements(
    old_package: str, new_package: str, req_files: Union[str, Sequence[str]] = REQUIREMENT_FILES_ALL
) -> None:
    """Replace one package by another with same version in given requirement files."""
    if isinstance(req_files, str):
        req_files = [req_files]
    for fname in req_files:
        with open(fname) as fopen:
            reqs = fopen.readlines()
        reqs = _replace_package_name(reqs, old_package, new_package)
        with open(fname, "w") as fw:
            fw.writelines(reqs)
