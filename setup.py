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
"""This is the main and only one setup entry point for installing each package as stand-alone as well as joint
installation for all packages.

There are considered three main scenarios for installing this project:

1. Using PyPI registry when you can install `pytorch-lightning`, `lightning-app`, etc. or `lightning` for all.

2. Installation from source code after cloning repository.
    In such case we recommend to use command `pip install .` or `pip install -e .` for development version
     (development ver. do not copy python files to your pip file system, just create links, so you can edit here)
    In case you want to install just one package you need to export env. variable before calling `pip`

     - for `pytorch-lightning` use `export PACKAGE_NAME=pytorch ; pip install .`
     - for `lightning-app` use `export PACKAGE_NAME=app ; pip install .`

3. Building packages as sdist or binary wheel and installing or publish to PyPI afterwords you use command
    `python setup.py sdist` or `python setup.py bdist_wheel` accordingly.
   In case you want to build just a particular package you would use exporting env. variable as above:
   `export PACKAGE_NAME=pytorch|app ; python setup.py sdist bdist_wheel`

4. Automated releasing with GitHub action is natural extension of 3) is composed of three consecutive steps:
    a) determine which packages shall be released based on version increment in `__version__.py` and eventually
     compared against PyPI registry
    b) with a parameterization build desired packages in to standard `dist/` folder
    c) validate packages and publish to PyPI


| Installation   | PIP version *       | Pkg version **  |
| -------------- | ------------------- | --------------- |
| source         | calendar + branch   | semantic        |
| PyPI           | semantic            | semantic        |

* shown version while calling `pip list | grep lightning`
** shown version in python `from <pytorch_lightning|lightning_app> import __version__`
"""
import os
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

from setuptools import setup

_PACKAGE_NAME = os.environ.get("PACKAGE_NAME", "")
_PACKAGE_MAPPING = {"pytorch": "pytorch_lightning", "app": "lightning_app"}
_REAL_PKG_NAME = _PACKAGE_MAPPING.get(_PACKAGE_NAME, _PACKAGE_NAME)
# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
_PATH_ROOT = os.path.dirname(__file__)
_PATH_SETUP = os.path.join(_PATH_ROOT, "src", _REAL_PKG_NAME or "lightning", "__setup__.py")


# Hardcode the env variable from time of package creation, otherwise it fails during installation
with open(__file__) as fp:
    lines = fp.readlines()
for i, ln in enumerate(lines):
    if ln.startswith("_PACKAGE_NAME = "):
        lines[i] = f'_PACKAGE_NAME = "{_PACKAGE_NAME}"{os.linesep}'
with open(__file__, "w") as fp:
    fp.writelines(lines)


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


# https://packaging.python.org/discussions/install-requires-vs-requirements /
# keep the meta-data here for simplicity in reading this file... it's not obvious
# what happens and to non-engineers they won't know to look in init ...
# the goal of the project is simplicity for researchers, don't want to add too much
# engineer specific practices
if __name__ == "__main__":
    _SETUP_TOOLS = _load_py_module(name="setup_tools", location=os.path.join(".actions", "setup_tools.py"))
    for lit_name, pkg_name in _PACKAGE_MAPPING.items():
        _SETUP_TOOLS.create_meta_package(os.path.join(_PATH_ROOT, "src"), pkg_name, lit_name)
    _SETUP_MODULE = _load_py_module(name="pkg_setup", location=_PATH_SETUP)
    _SETUP_MODULE._adjust_manifest(pkg_name=_REAL_PKG_NAME)
    setup(**_SETUP_MODULE._setup_args(pkg_name=_REAL_PKG_NAME))
