# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import glob
import os.path
from pprint import pprint
from typing import Sequence

REQUIREMENT_ROOT = "requirements.txt"
REQUIREMENT_FILES_ALL: list = glob.glob(os.path.join("requirements", "*.txt"))
REQUIREMENT_FILES_ALL += glob.glob(os.path.join("requirements", "**", "*.txt"), recursive=True)
if os.path.isfile(REQUIREMENT_ROOT):
    REQUIREMENT_FILES_ALL += [REQUIREMENT_ROOT]


def prune_pkgs_in_requirements(packages: Sequence[str], req_files: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
    """Remove some packages from given requirement files."""
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
    req = open(fname).read().replace(">=", "==")
    open(fname, "w").write(req)


def replace_oldest_ver(req_files: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
    """Replace the min package version by fixed one."""
    for fname in req_files:
        _replace_min(fname)
