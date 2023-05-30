import os
from typing import List

from setuptools import find_packages, setup

_PROJECT_ROOT = os.path.dirname(__file__)


def _load_requirements(path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#") -> List[str]:
    """Load requirements from a file."""
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        # skip index url
        if ln.startswith("--extra-index-url"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(
    name="{{ app_name }}",
    version="0.0.1",
    packages=find_packages(exclude=["ui"]),
    python_requires=">=3.8",
)
