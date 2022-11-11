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
     - for `lightning-lite` use `export PACKAGE_NAME=lite ; pip install .`
     - for `lightning-app` use `export PACKAGE_NAME=app ; pip install .`

3. Building packages as sdist or binary wheel and installing or publish to PyPI afterwords you use command
    `python setup.py sdist` or `python setup.py bdist_wheel` accordingly.
   In case you want to build just a particular package you would use exporting env. variable as above:
   `export PACKAGE_NAME=pytorch|app|lite ; python setup.py sdist bdist_wheel`

4. Automated releasing with GitHub action is natural extension of 3) is composed of three consecutive steps:
    a) determine which packages shall be released based on version increment in `__version__.py` and eventually
     compared against PyPI registry
    b) with a parameterization build desired packages in to standard `dist/` folder
    c) validate packages and publish to PyPI
"""
import os
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

from setuptools import setup

_PACKAGE_NAME = os.environ.get("PACKAGE_NAME")
_PACKAGE_MAPPING = {
    "lightning": "lightning",
    "pytorch": "pytorch_lightning",
    "app": "lightning_app",
    "lite": "lightning_lite",
}
# https://packaging.python.org/guides/single-sourcing-package-version/
# http://blog.ionelmc.ro/2014/05/25/python-packaging/
_PATH_ROOT = os.path.dirname(__file__)
_PATH_SRC = os.path.join(_PATH_ROOT, "src")
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")
_FREEZE_REQUIREMENTS = bool(int(os.environ.get("FREEZE_REQUIREMENTS", 0)))


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


if __name__ == "__main__":
    setup_tools = _load_py_module(name="setup_tools", location=os.path.join(_PATH_ROOT, ".actions", "setup_tools.py"))
    assistant = _load_py_module(name="assistant", location=os.path.join(_PATH_ROOT, ".actions", "assistant.py"))

    package_to_install = _PACKAGE_NAME or "lightning"
    print(f"Installing the {package_to_install} package")  # requires `-v` to appear
    if package_to_install == "lightning":  # install everything
        # merge all requirements files
        setup_tools._load_aggregate_requirements(_PATH_REQUIRE, _FREEZE_REQUIREMENTS)
        # replace imports and copy the code
        assistant.create_mirror_package(_PATH_SRC, _PACKAGE_MAPPING)
    elif package_to_install not in _PACKAGE_MAPPING:
        raise ValueError(f"Unexpected package name: {_PACKAGE_NAME}. Possible choices are: {list(_PACKAGE_MAPPING)}")

    # if `_PACKAGE_NAME` is not set, iterate over all possible packages until we find one that can be installed.
    # this is useful for installing existing wheels, as the user wouldn't set this environment variable, but the wheel
    # should have included only the relevant files of the package to install
    possible_packages = _PACKAGE_MAPPING.values() if _PACKAGE_NAME is None else [_PACKAGE_MAPPING[_PACKAGE_NAME]]
    for pkg in possible_packages:
        pkg_setup = os.path.join(_PATH_SRC, pkg, "__setup__.py")
        if os.path.exists(pkg_setup):
            print(f"{pkg_setup} exists. Running `setuptools.setup`")
            setup_module = _load_py_module(name=f"{pkg}_setup", location=pkg_setup)
            setup_module._adjust_manifest(pkg_name=pkg)
            setup_args = setup_module._setup_args(pkg_name=pkg)
            setup(**setup_args)
            break
    else:
        raise RuntimeError(f"Something's wrong, no package was installed. Package name: {_PACKAGE_NAME}")
