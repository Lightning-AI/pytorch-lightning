import glob
import os
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

from pkg_resources import parse_requirements
from setuptools import find_packages

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "lightning_fabric")
_PATH_REQUIREMENTS = os.path.join("requirements", "fabric")
_FREEZE_REQUIREMENTS = bool(int(os.environ.get("FREEZE_REQUIREMENTS", 0)))


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


def _prepare_extras() -> Dict[str, Any]:
    assistant = _load_assistant()
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
    # Define package extras. These are only installed if you specify them.
    # From remote, use like `pip install pytorch-lightning[dev, docs]`
    # From local copy of repo, use like `pip install ".[dev, docs]"`
    common_args = dict(path_dir=_PATH_REQUIREMENTS, unfreeze="none" if _FREEZE_REQUIREMENTS else "all")
    req_files = [Path(p) for p in glob.glob(os.path.join(_PATH_REQUIREMENTS, "*.txt"))]
    extras = {
        p.stem: assistant.load_requirements(file_name=p.name, **common_args)
        for p in req_files
        if p.name not in ("docs.txt", "devel.txt", "base.txt")
    }
    for req in parse_requirements(extras["strategies"]):
        extras[req.key] = [str(req)]
    extras["all"] = extras["strategies"] + extras["examples"]
    extras["dev"] = extras["all"] + extras["test"]
    return extras


def _setup_args() -> Dict[str, Any]:
    assistant = _load_assistant()
    about = _load_py_module("about", os.path.join(_PACKAGE_ROOT, "__about__.py"))
    version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))
    long_description = assistant.load_readme_description(
        _PACKAGE_ROOT, homepage=about.__homepage__, version=version.version
    )

    return dict(
        name="lightning-fabric",
        version=version.version,
        description=about.__docs__,
        author=about.__author__,
        author_email=about.__author_email__,
        url=about.__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=about.__license__,
        packages=find_packages(where="src", include=["lightning_fabric", "lightning_fabric.*"]),
        package_dir={"": "src"},
        long_description=long_description,
        long_description_content_type="text/markdown",
        include_package_data=True,
        zip_safe=False,
        keywords=["deep learning", "pytorch", "AI"],
        python_requires=">=3.7",
        setup_requires=["wheel"],
        install_requires=assistant.load_requirements(
            _PATH_REQUIREMENTS, unfreeze="none" if _FREEZE_REQUIREMENTS else "all"
        ),
        extras_require=_prepare_extras(),
        project_urls={
            "Bug Tracker": "https://github.com/Lightning-AI/lightning/issues",
            "Documentation": "https://pytorch-lightning.rtfd.io/en/latest/",
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
