import os.path
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Any, Dict

from pkg_resources import parse_requirements
from setuptools import find_packages

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "pytorch_lightning")
_PATH_REQUIREMENTS = os.path.join("requirements", "pytorch")
_FREEZE_REQUIREMENTS = bool(int(os.environ.get("FREEZE_REQUIREMENTS", 0)))


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


def _prepare_extras(**kwargs: Any) -> Dict[str, Any]:
    _path_setup_tools = os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py")
    _setup_tools = _load_py_module("setup_tools", _path_setup_tools)
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
    # Define package extras. These are only installed if you specify them.
    # From remote, use like `pip install pytorch-lightning[dev, docs]`
    # From local copy of repo, use like `pip install ".[dev, docs]"`
    common_args = dict(path_dir=_PATH_REQUIREMENTS, unfreeze=not _FREEZE_REQUIREMENTS)
    extras = {
        # 'docs': load_requirements(file_name='docs.txt'),
        "examples": _setup_tools.load_requirements(file_name="examples.txt", **common_args),
        "loggers": _setup_tools.load_requirements(file_name="loggers.txt", **common_args),
        "extra": _setup_tools.load_requirements(file_name="extra.txt", **common_args),
        "strategies": _setup_tools.load_requirements(file_name="strategies.txt", **common_args),
        "test": _setup_tools.load_requirements(file_name="test.txt", **common_args),
    }
    for req in parse_requirements(extras["strategies"]):
        extras[req.key] = [str(req)]
    extras["dev"] = extras["extra"] + extras["loggers"] + extras["test"]
    extras["all"] = extras["dev"] + extras["examples"] + extras["strategies"]  # + extras['docs']
    return extras


def _adjust_manifest(**__: Any) -> None:
    manifest_path = os.path.join(_PROJECT_ROOT, "MANIFEST.in")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as fp:
        lines = fp.readlines()
    lines += [
        "recursive-exclude src *.md" + os.linesep,
        "recursive-exclude requirements *.txt" + os.linesep,
        "recursive-include src/pytorch_lightning *.md" + os.linesep,
        "recursive-include requirements/pytorch *.txt" + os.linesep,
        "include src/pytorch_lightning/py.typed" + os.linesep,  # marker file for PEP 561
    ]
    with open(manifest_path, "w") as fp:
        fp.writelines(lines)


def _setup_args(**__: Any) -> Dict[str, Any]:
    _path_setup_tools = os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py")
    _setup_tools = _load_py_module("setup_tools", _path_setup_tools)
    _about = _load_py_module("about", os.path.join(_PACKAGE_ROOT, "__about__.py"))
    _version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))
    _long_description = _setup_tools.load_readme_description(
        _PACKAGE_ROOT, homepage=_about.__homepage__, version=_version.version
    )
    return dict(
        name="pytorch-lightning",
        version=_version.version,  # todo: consider using date version + branch for installation from source
        description=_about.__docs__,
        author=_about.__author__,
        author_email=_about.__author_email__,
        url=_about.__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=_about.__license__,
        packages=find_packages(where="src", include=["pytorch_lightning", "pytorch_lightning.*"]),
        package_dir={"": "src"},
        include_package_data=True,
        long_description=_long_description,
        long_description_content_type="text/markdown",
        zip_safe=False,
        keywords=["deep learning", "pytorch", "AI"],
        python_requires=">=3.7",
        setup_requires=[],
        install_requires=_setup_tools.load_requirements(_PATH_REQUIREMENTS, unfreeze=not _FREEZE_REQUIREMENTS),
        extras_require=_prepare_extras(),
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
