#!/usr/bin/env python
# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import List

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
_PATH_ROOT = os.path.dirname(__file__)
_PATH_REQUIREMENTS = os.path.join(_PATH_ROOT, "requirements", "pytorch")
_PATH_PL_SRC = os.path.join(_PATH_ROOT, "src", "pytorch_lightning")


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _load_requirements(
    path_dir: str, file_name: str = "base.txt", comment_char: str = "#", unfreeze: bool = True
) -> List[str]:
    """Load requirements from a file.

    >>> _load_requirements(_PATH_REQUIREMENTS)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        comment = ""
        if comment_char in ln:
            comment = ln[ln.index(comment_char) :]
            ln = ln[: ln.index(comment_char)]
        req = ln.strip()
        # skip directly installed dependencies
        if not req or req.startswith("http") or "@http" in req:
            continue
        # remove version restrictions unless they are strict
        if unfreeze and "<" in req and "strict" not in comment:
            req = re.sub(r",? *<=? *[\d\.\*]+", "", req).strip()
        reqs.append(req)
    return reqs


def _load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as decribtion.

    >>> _load_readme_description(_PATH_ROOT, "", "")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '<div align="center">...'
    """
    path_readme = os.path.join(path_dir, "README.md")
    text = open(path_readme, encoding="utf-8").read()

    # drop images from readme
    text = text.replace("![PT to PL](docs/source/_static/images/general/pl_quick_start_full_compressed.gif)", "")

    # https://github.com/Lightning-AI/lightning/raw/master/docs/source/_static/images/lightning_module/pt_to_pl.png
    github_source_url = os.path.join(homepage, "raw", version)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we reger some other sources with particular path to docs
    text = text.replace("docs/source/_static/", f"{os.path.join(github_source_url, 'docs/source/_static/')}")

    # readthedocs badge
    text = text.replace("badge/?version=stable", f"badge/?version={version}")
    text = text.replace("pytorch-lightning.readthedocs.io/en/stable/", f"pytorch-lightning.readthedocs.io/en/{version}")
    # codecov badge
    text = text.replace("/branch/master/graph/badge.svg", f"/release/{version}/graph/badge.svg")
    # replace github badges for release ones
    text = text.replace("badge.svg?branch=master&event=push", f"badge.svg?tag={version}")
    # Azure...
    text = text.replace("?branchName=master", f"?branchName=refs%2Ftags%2F{version}")
    text = re.sub(r"\?definitionId=\d+&branchName=master", f"?definitionId=2&branchName=refs%2Ftags%2F{version}", text)

    skip_begin = r"<!-- following section will be skipped from PyPI description -->"
    skip_end = r"<!-- end skipping PyPI description -->"
    # todo: wrap content as commented description
    text = re.sub(rf"{skip_begin}.+?{skip_end}", "<!--  -->", text, flags=re.IGNORECASE + re.DOTALL)

    # # https://github.com/Borda/pytorch-lightning/releases/download/1.1.0a6/codecov_badge.png
    # github_release_url = os.path.join(homepage, "releases", "download", version)
    # # download badge and replace url with local file
    # text = _parse_for_badge(text, github_release_url)
    return text


_ABOUT_MODULE = _load_py_module(name="about", location=os.path.join(_PATH_PL_SRC, "__about__.py"))

# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
# Define package extras. These are only installed if you specify them.
# From remote, use like `pip install pytorch-lightning[dev, docs]`
# From local copy of repo, use like `pip install ".[dev, docs]"`
extras = {
    # 'docs': load_requirements(file_name='docs.txt'),
    "examples": _load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="examples.txt"),
    "loggers": _load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="loggers.txt"),
    "extra": _load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="extra.txt"),
    "strategies": _load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="strategies.txt"),
    "test": _load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="test.txt"),
}
for req in parse_requirements(extras["strategies"]):
    extras[req.key] = [str(req)]
extras["dev"] = extras["extra"] + extras["loggers"] + extras["test"]
extras["all"] = extras["dev"] + extras["examples"] + extras["strategies"]  # + extras['docs']

long_description = _load_readme_description(
    _PATH_ROOT, homepage=_ABOUT_MODULE.__homepage__, version=_ABOUT_MODULE.__version__
)

# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
if __name__ == "__main__":
    setup(
        name="pytorch-lightning",
        version=_ABOUT_MODULE.__version__,
        description=_ABOUT_MODULE.__docs__,
        author=_ABOUT_MODULE.__author__,
        author_email=_ABOUT_MODULE.__author_email__,
        url=_ABOUT_MODULE.__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=_ABOUT_MODULE.__license__,
        # todo: temp disable installing apps from source
        packages=find_packages(where="src", exclude=["lightning", "lightning_app", "lightning_app.*"]),
        package_dir={"": "src"},
        include_package_data=True,
        long_description=long_description,
        long_description_content_type="text/markdown",
        zip_safe=False,
        keywords=["deep learning", "pytorch", "AI"],
        python_requires=">=3.7",
        setup_requires=[],
        install_requires=_load_requirements(_PATH_REQUIREMENTS),
        extras_require=extras,
        project_urls={
            "Bug Tracker": "https://github.com/Lightning-AI/lightning/issues",
            "Documentation": "https://pytorch-lightning.rtfd.io/en/latest/",
            "Source Code": "https://github.com/Lightning-AI/lightning",
        },
        classifiers=[
            "Environment :: Console",
            "Natural Language :: English",
            "Development Status :: 5 - Production/Stable",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: Scientific/Engineering :: Information Analysis",
            # Pick your license as you wish
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            # Specify the Python versions you support here.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],
    )
