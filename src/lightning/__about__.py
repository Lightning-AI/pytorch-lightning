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

import os.path
import time
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

from setuptools import find_packages

__author__ = "Lightning AI et al."
__author_email__ = "pytorch@lightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2018-{time.strftime('%Y')}, {__author__}."
__homepage__ = "https://github.com/Lightning-AI/pytorch-lightning"
__docs_url__ = "https://pytorch-lightning.readthedocs.io/en/stable/"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = (
    "Use Lightning Apps to build everything from production-ready, multi-cloud ML systems to simple research demos."
)
__long_doc__ = """
What is it?
-----------

TBD @eden
"""  # TODO

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__docs_url__",
    "__homepage__",
    "__license__",
]

_PACKAGE_ROOT = os.path.dirname(__file__)
_SOURCE_ROOT = os.path.dirname(_PACKAGE_ROOT)
_PROJECT_ROOT = os.path.dirname(_SOURCE_ROOT)


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


_setup_tools = _load_py_module("setup_tools", os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py"))
__version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))


_long_description = _setup_tools.load_readme_description(
    _PROJECT_ROOT, homepage=__homepage__, version=__version.version
)


def _setup_args():
    # todo: consider invaliding some additional arguments from packages, for example if include data or safe to zip
    return dict(
        name="lightning",
        version=__version.version,  # todo: consider adding branch for installation from source
        description=__docs__,
        author=__author__,
        author_email=__author_email__,
        url=__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=__license__,
        packages=find_packages(
            where="src", include=["lightning", "lightning.*"]
        ),  # todo: if install from source include all package and remove them from requirements
        package_dir={"": "src"},
        long_description=_long_description,
        long_description_content_type="text/markdown",
        keywords=["deep learning", "pytorch", "AI"],  # todo: aggregate tags from all packages
        python_requires=">=3.7",  # todo: take the lowes based on all packages
        setup_requires=[],
        install_requires=["pytorch-lightning==1.6.*", "lightning-app==0.5.*"],  # todo: generate this list automatically
        extras_require={},  # todo: consider porting all other packages extras with prefix
        project_urls={
            "Bug Tracker": "https://github.com/Lightning-AI/lightning/issues",
            "Source Code": "https://github.com/Lightning-AI/lightning",
        },
        classifiers=[
            "Environment :: Console",
            "Natural Language :: English",
            "Development Status :: 5 - Production/Stable",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            # Pick your license as you wish
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            # Specify the Python versions you support here.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
        ],  # todo: consider aggregation/union of tags from particular packages
    )
