import os.path
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Any, Dict

from setuptools import find_packages

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "lightning")
_PATH_REQUIREMENTS = os.path.join("requirements")
_FREEZE_REQUIREMENTS = bool(int(os.environ.get("FREEZE_REQUIREMENTS", 0)))


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


_SETUP_TOOLS = _load_py_module("setup_tools", os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py"))


def _adjust_manifest(**kwargs: Any) -> None:
    # todo: consider rather aggregation of particular manifest adjustments
    manifest_path = os.path.join(_PROJECT_ROOT, "MANIFEST.in")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as fp:
        lines = [ln.rstrip() for ln in fp.readlines()]
    lines += [
        "recursive-include src/lightning *.md",
        "recursive-include requirements *.txt",
        "recursive-include src/lightning/app/ui *",
        "recursive-include src/lightning/cli/*-template *",  # Add templates as build-in
        "include src/lightning/app/components/serve/catimage.png" + os.linesep,
        # fixme: this is strange, this shall work with setup find package - include
        "prune src/lightning_app",
        "prune src/lightning_lite",
        "prune src/pytorch_lightning",
    ]
    with open(manifest_path, "w") as fp:
        fp.writelines([ln + os.linesep for ln in lines])


def _setup_args(**kwargs: Any) -> Dict[str, Any]:
    _about = _load_py_module("about", os.path.join(_PACKAGE_ROOT, "__about__.py"))
    _version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))
    _long_description = _SETUP_TOOLS.load_readme_description(
        _PROJECT_ROOT, homepage=_about.__homepage__, version=_version.version
    )
    _include_pkgs = ["lightning", "lightning.*"]

    # TODO: consider invaliding some additional arguments from packages, for example if include data or safe to zip

    # TODO: remove this once lightning-ui package is ready as a dependency
    _SETUP_TOOLS._download_frontend(os.path.join(_SOURCE_ROOT, "lightning", "app"))

    return dict(
        name="lightning",
        version=_version.version,  # todo: consider adding branch for installation from source
        description=_about.__docs__,
        author=_about.__author__,
        author_email=_about.__author_email__,
        url=_about.__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=_about.__license__,
        packages=find_packages(where="src", include=_include_pkgs),
        package_dir={"": "src"},
        long_description=_long_description,
        long_description_content_type="text/markdown",
        include_package_data=True,
        zip_safe=False,
        keywords=["deep learning", "pytorch", "AI"],  # todo: aggregate tags from all packages
        python_requires=">=3.7",  # todo: take the lowes based on all packages
        entry_points={
            "console_scripts": [
                "lightning = lightning.app.cli.lightning_cli:main",
            ],
        },
        setup_requires=[],
        install_requires=_SETUP_TOOLS.load_requirements(_PATH_REQUIREMENTS, unfreeze="all"),
        extras_require={},  # todo: consider porting all other packages extras with prefix
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
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            # Specify the Python versions you support here.
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
        ],  # todo: consider aggregation/union of tags from particular packages
    )
