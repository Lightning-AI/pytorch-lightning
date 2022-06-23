import os.path
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

from setuptools import find_packages

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "lightning")


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


def _adjust_manifest():
    # todo
    pass


def _setup_args():
    _path_setup_tools = os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py")
    _setup_tools = _load_py_module("setup_tools", _path_setup_tools)
    _about = _load_py_module("about", os.path.join(_PACKAGE_ROOT, "__about__.py"))
    _version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))
    _long_description = _setup_tools.load_readme_description(
        _PROJECT_ROOT, homepage=_about.__homepage__, version=_version.version
    )
    # todo: consider invaliding some additional arguments from packages, for example if include data or safe to zip
    return dict(
        name="lightning",
        version=_version.version,  # todo: consider adding branch for installation from source
        description=_about.__docs__,
        author=_about.__author__,
        author_email=_about.__author_email__,
        url=_about.__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=_about.__license__,
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
