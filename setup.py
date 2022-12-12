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
   In case you want to build just a particular package you want to set an environment variable:
   `PACKAGE_NAME=lightning|pytorch|app|lite python setup.py sdist|bdist_wheel`

4. Automated releasing with GitHub action is natural extension of 3) is composed of three consecutive steps:
    a) determine which packages shall be released based on version increment in `__version__.py` and eventually
     compared against PyPI registry
    b) with a parameterization build desired packages in to standard `dist/` folder
    c) validate packages and publish to PyPI
"""
import contextlib
import os
import tempfile
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Generator, Optional

import setuptools
import setuptools.command.egg_info

_PACKAGE_NAME = os.environ.get("PACKAGE_NAME", "lightning")
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


def _named_temporary_file(directory: Optional[str] = None) -> str:
    # `tempfile.NamedTemporaryFile` has issues in Windows
    # https://github.com/deepchem/deepchem/issues/707#issuecomment-556002823
    if directory is None:
        directory = tempfile.gettempdir()
    return os.path.join(directory, os.urandom(24).hex())


@contextlib.contextmanager
def _set_manifest_path(manifest_dir: str, aggregate: bool = False) -> Generator:
    if aggregate:
        # aggregate all MANIFEST.in contents into a single temporary file
        manifest_path = _named_temporary_file(manifest_dir)
        mapping = _PACKAGE_MAPPING.copy()
        del mapping["lightning"]
        lines = ["include src/lightning/version.info\n"]
        for pkg in mapping.values():
            pkg_path = os.path.join(_PATH_SRC, pkg)
            if not os.path.exists(pkg_path):
                # this function is getting called with `pip install .` and `pip install *.tar.gz`, however, it only
                # should be called with the former. i haven't found a way to differentiate the two so this is the hacky
                # solution to avoid an error
                print(f"{pkg_path!r} does not exist")
                yield
                return
            with open(os.path.join(pkg_path, "MANIFEST.in")) as fh:
                lines.extend(fh.readlines())
        # convert lightning_foo to lightning/foo
        for new, old in mapping.items():
            lines = [line.replace(old, f"lightning/{new}") for line in lines]
        with open(manifest_path, mode="w") as fp:
            fp.writelines(lines)
    else:
        manifest_path = os.path.join(manifest_dir, "MANIFEST.in")
        assert os.path.exists(manifest_path)
    # avoid error: setup script specifies an absolute path
    manifest_path = os.path.relpath(manifest_path, _PATH_ROOT)
    print("Set manifest path to", manifest_path)
    setuptools.command.egg_info.manifest_maker.template = manifest_path
    yield
    # cleanup
    setuptools.command.egg_info.manifest_maker.template = "MANIFEST.in"
    if aggregate:
        os.remove(manifest_path)


if __name__ == "__main__":
    setup_tools = _load_py_module(name="setup_tools", location=os.path.join(_PATH_ROOT, ".actions", "setup_tools.py"))
    assistant = _load_py_module(name="assistant", location=os.path.join(_PATH_ROOT, ".actions", "assistant.py"))

    if os.path.exists(_PATH_SRC):
        # copy the version information to all packages
        setup_tools.distribute_version(_PATH_SRC)

    package_to_install = _PACKAGE_NAME or "lightning"
    print(f"Installing the {package_to_install} package")  # requires `-v` to appear
    is_wheel_install = "PEP517_BUILD_BACKEND" in os.environ
    print("is_wheel_install:", is_wheel_install)
    if package_to_install not in _PACKAGE_MAPPING or (not is_wheel_install and _PACKAGE_NAME is None):
        raise ValueError(f"Unexpected package name: {_PACKAGE_NAME}. Possible choices are: {list(_PACKAGE_MAPPING)}")
    is_wheel_install &= _PACKAGE_NAME is None

    if package_to_install == "lightning":  # install everything
        # merge all requirements files
        setup_tools._load_aggregate_requirements(_PATH_REQUIRE, _FREEZE_REQUIREMENTS)
        # replace imports and copy the code
        assistant.create_mirror_package(_PATH_SRC, _PACKAGE_MAPPING)

    # if it's a wheel install (hence _PACKAGE_NAME should not be set), iterate over all possible packages until we find
    # one that can be installed. the wheel should have included only the relevant files of the package to install
    possible_packages = _PACKAGE_MAPPING.values() if is_wheel_install else [_PACKAGE_MAPPING[_PACKAGE_NAME]]
    for pkg in possible_packages:
        pkg_path = os.path.join(_PATH_SRC, pkg)
        pkg_setup = os.path.join(pkg_path, "__setup__.py")
        if os.path.exists(pkg_setup):
            print(f"{pkg_setup} exists. Running `setuptools.setup`")
            setup_module = _load_py_module(name=f"{pkg}_setup", location=pkg_setup)
            setup_args = setup_module._setup_args()
            is_main_pkg = pkg == "lightning"
            if is_wheel_install and not is_main_pkg:
                setuptools.setup(**setup_args)
            else:
                # we are installing from source, set the correct manifest path
                with _set_manifest_path(pkg_path, aggregate=is_main_pkg):
                    setuptools.setup(**setup_args)
            break
    else:
        raise RuntimeError(f"Something's wrong, no package was installed. Package name: {_PACKAGE_NAME}")
