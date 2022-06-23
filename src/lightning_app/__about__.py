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
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

from setuptools import find_packages

# __version__ = "0.5.1"
__author__ = "PyTorchLightning et al."
__author_email__ = "name@pytorchlightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2021-2022, {__author__}."
__homepage__ = "https://github.com/PyTorchLightning/lightning"
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
    "__homepage__",
    "__license__",
]

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "lightning_app")
_PATH_REQUIREMENTS = os.path.join("requirements", "app")


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


_setup_tools = _load_py_module(
    "setup_tools",
    os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py"),
)
__version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))


_long_description = _setup_tools.load_readme_description(
    _PACKAGE_ROOT, homepage=__homepage__, version=__version.version
)


def _prepare_extras():
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
    # Define package extras. These are only installed if you specify them.
    # From remote, use like `pip install pytorch-lightning[dev, docs]`
    # From local copy of repo, use like `pip install ".[dev, docs]"`
    extras = {
        # 'docs': load_requirements(file_name='docs.txt'),
        "cloud": _setup_tools.load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="cloud.txt"),
        "ui": _setup_tools.load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="ui.txt"),
        "test": _setup_tools.load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="test.txt"),
    }
    extras["dev"] = extras["cloud"] + extras["ui"] + extras["test"]  # + extras['docs']
    extras["all"] = extras["cloud"] + extras["ui"]
    return extras


def _adjust_manifest():
    manifest_path = os.path.join(_PROJECT_ROOT, "MANIFEST.in")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as fp:
        lines = fp.readlines()
    lines += [
        "recursive-include src/lightning_app *.md" + os.linesep,
        "recursive-include requirements/app *.txt" + os.linesep,
    ]
    with open(manifest_path, "w") as fp:
        fp.writelines(lines)


def _setup_args():
    # TODO: at this point we need to download the UI to the package
    return dict(
        name="lightning-app",
        version=__version.version,  # todo: consider using date version + branch for installation from source
        description=__docs__,
        author=__author__,
        author_email=__author_email__,
        url=__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=__license__,
        packages=find_packages(where="src", include=["lightning_app", "lightning_app.*"]),
        package_dir={"": "src"},
        long_description=_long_description,
        long_description_content_type="text/markdown",
        include_package_data=True,
        zip_safe=False,
        keywords=["deep learning", "pytorch", "AI"],
        python_requires=">=3.7",
        entry_points={
            "console_scripts": [
                "lightning = lightning_app.cli.lightning_cli:main",
            ],
        },
        setup_requires=["wheel"],
        install_requires=_setup_tools.load_requirements(_PATH_REQUIREMENTS),
        extras_require=_prepare_extras(),
        project_urls={
            "Bug Tracker": "https://github.com/Lightning-AI/lightning/issues",
            "Documentation": "https://lightning.ai/lightning-docs",
            "Source Code": "https://github.com/Lightning-AI/lightning",
        },
        classifiers=[
            "Environment :: Console",
            "Natural Language :: English",
            # How mature is this project? Common values are
            #   3 - Alpha, 4 - Beta, 5 - Production/Stable
            "Development Status :: 4 - Beta",
            # Indicate who your project is intended for
            "Intended Audience :: Developers",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Topic :: Scientific/Engineering :: Information Analysis",
            # Pick your license as you wish
            # 'License :: OSI Approved :: BSD License',
            "Operating System :: OS Independent",
            # Specify the Python versions you support here. In particular, ensure
            # that you indicate whether you support Python 2, Python 3 or both.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],
    )
