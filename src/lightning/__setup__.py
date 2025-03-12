import glob
import os.path
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any

from setuptools import find_namespace_packages

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "lightning")
_PATH_REQUIREMENTS = os.path.join(_PROJECT_ROOT, "requirements")
_FREEZE_REQUIREMENTS = os.environ.get("FREEZE_REQUIREMENTS", "0").lower() in ("1", "true")


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


_ASSISTANT = _load_py_module(name="assistant", location=os.path.join(_PROJECT_ROOT, ".actions", "assistant.py"))


def _prepare_extras() -> dict[str, Any]:
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
    # Define package extras. These are only installed if you specify them.
    # From remote, use like `pip install "lightning[dev, docs]"`
    # From local copy of repo, use like `pip install ".[dev, docs]"`
    req_files = [Path(p) for p in glob.glob(os.path.join(_PATH_REQUIREMENTS, "*", "*.txt"))]
    common_args = {"unfreeze": "none" if _FREEZE_REQUIREMENTS else "major"}
    # per-project extras
    extras = {
        f"{p.parent.name}-{p.stem}": _ASSISTANT.load_requirements(file_name=p.name, path_dir=p.parent, **common_args)
        for p in req_files
        if p.name not in ("docs.txt", "base.txt") and not p.parent.name.startswith("_")
    }

    # project specific extras groups
    extras["fabric-all"] = extras["fabric-strategies"] + extras["fabric-examples"]
    extras["fabric-dev"] = extras["fabric-all"] + extras["fabric-test"]
    extras["pytorch-all"] = extras["pytorch-extra"] + extras["pytorch-strategies"] + extras["pytorch-examples"]
    extras["pytorch-dev"] = extras["pytorch-all"] + extras["pytorch-test"]

    # merge per-project extras of the same category
    for extra in list(extras):
        name = "-".join(extra.split("-")[1:])
        extras[name] = extras.get(name, []) + extras[extra]

    # drop quasi base the req. file has the same name sub-package
    for k in list(extras.keys()):
        kk = k.split("-")
        if not (len(kk) == 2 and kk[0] == kk[1]):
            continue
        extras[kk[0]] = list(extras[k])
        del extras[k]
    extras = {name: sorted(set(reqs)) for name, reqs in extras.items()}
    print("The extras are: ", extras)
    return extras


def _setup_args() -> dict[str, Any]:
    about = _load_py_module("about", os.path.join(_PACKAGE_ROOT, "__about__.py"))
    version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))
    long_description = _ASSISTANT.load_readme_description(
        _PROJECT_ROOT, homepage=about.__homepage__, version=version.version
    )

    # TODO: consider invaliding some additional arguments from packages, for example if include data or safe to zip

    install_requires = _ASSISTANT.load_requirements(
        _PATH_REQUIREMENTS, unfreeze="none" if _FREEZE_REQUIREMENTS else "major"
    )
    # toto: remove when we realize that this is making confusion as cross pkg import is not fully compatible
    install_requires += ["pytorch-lightning"]

    return {
        "name": "lightning",
        "version": version.version,
        "description": about.__docs__,
        "author": about.__author__,
        "author_email": about.__author_email__,
        "url": about.__homepage__,
        "download_url": "https://github.com/Lightning-AI/lightning",
        "license": about.__license__,
        "packages": find_namespace_packages(where="src", include=["lightning", "lightning.*"]),
        "package_dir": {"": "src"},
        "long_description": long_description,
        "long_description_content_type": "text/markdown",
        "include_package_data": True,
        "zip_safe": False,
        "keywords": ["deep learning", "pytorch", "AI"],
        "python_requires": ">=3.9",
        "entry_points": {
            "console_scripts": [
                "fabric = lightning.fabric.cli:_main",
            ],
        },
        "setup_requires": [],
        "install_requires": install_requires,
        "extras_require": _prepare_extras(),
        "project_urls": {
            "Bug Tracker": "https://github.com/Lightning-AI/pytorch-lightning/issues",
            "Documentation": "https://lightning.ai/lightning-docs",
            "Source Code": "https://github.com/Lightning-AI/lightning",
        },
        "classifiers": [
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
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            # Specify the Python versions you support here.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
        ],  # todo: consider aggregation/union of tags from particular packages
    }
