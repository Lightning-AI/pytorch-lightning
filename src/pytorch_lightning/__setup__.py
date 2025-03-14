import glob
import os.path
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from pkg_resources import parse_requirements
from setuptools import find_packages

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "pytorch_lightning")
_PATH_REQUIREMENTS = os.path.join("requirements", "pytorch")
_FREEZE_REQUIREMENTS = os.environ.get("FREEZE_REQUIREMENTS", "0").lower() in ("1", "true")


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


def _load_assistant() -> ModuleType:
    location = os.path.join(_PROJECT_ROOT, ".actions", "assistant.py")
    return _load_py_module("assistant", location)


def _prepare_extras() -> dict[str, Any]:
    assistant = _load_assistant()
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
    # Define package extras. These are only installed if you specify them.
    # From remote, use like `pip install "pytorch-lightning[dev, docs]"`
    # From local copy of repo, use like `PACKAGE_NAME=pytorch pip install ".[dev, docs]"`
    common_args = {"path_dir": _PATH_REQUIREMENTS, "unfreeze": "none" if _FREEZE_REQUIREMENTS else "all"}
    req_files = [Path(p) for p in glob.glob(os.path.join(_PATH_REQUIREMENTS, "*.txt"))]
    extras = {
        p.stem: assistant.load_requirements(file_name=p.name, **common_args)
        for p in req_files
        if p.name not in ("docs.txt", "base.txt")
    }
    for req in parse_requirements(extras["strategies"]):
        extras[req.key] = [str(req)]
    extras["all"] = extras["extra"] + extras["strategies"] + extras["examples"]
    extras["dev"] = extras["all"] + extras["test"]  # + extras['docs']
    return extras


def _setup_args() -> dict[str, Any]:
    assistant = _load_assistant()
    about = _load_py_module("about", os.path.join(_PACKAGE_ROOT, "__about__.py"))
    version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))
    long_description = assistant.load_readme_description(
        _PACKAGE_ROOT, homepage=about.__homepage__, version=version.version
    )
    return {
        "name": "pytorch-lightning",
        "version": version.version,
        "description": about.__docs__,
        "author": about.__author__,
        "author_email": about.__author_email__,
        "url": about.__homepage__,
        "download_url": "https://github.com/Lightning-AI/lightning",
        "license": about.__license__,
        "packages": find_packages(
            where="src",
            include=[
                "pytorch_lightning",
                "pytorch_lightning.*",
                "lightning_fabric",
                "lightning_fabric.*",
            ],
        ),
        "package_dir": {"": "src"},
        "include_package_data": True,
        "long_description": long_description,
        "long_description_content_type": "text/markdown",
        "zip_safe": False,
        "keywords": ["deep learning", "pytorch", "AI"],
        "python_requires": ">=3.9",
        "setup_requires": ["wheel"],
        # TODO: aggregate pytorch and lite requirements as we include its source code directly in this package.
        # this is not a problem yet because lite's base requirements are all included in pytorch's base requirements
        "install_requires": assistant.load_requirements(
            _PATH_REQUIREMENTS, unfreeze="none" if _FREEZE_REQUIREMENTS else "all"
        ),
        "extras_require": _prepare_extras(),
        "project_urls": {
            "Bug Tracker": "https://github.com/Lightning-AI/pytorch-lightning/issues",
            "Documentation": "https://pytorch-lightning.rtfd.io/en/latest/",
            "Source Code": "https://github.com/Lightning-AI/lightning",
        },
        "classifiers": [
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
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
        ],
    }
