import os.path
import time
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

from pkg_resources import parse_requirements
from setuptools import find_packages

# __version__ = "1.7.0dev"
__author__ = "Lightning AI et al."
__author_email__ = "pytorch@lightning.ai"
__license__ = "Apache-2.0"
__copyright__ = f"Copyright (c) 2018-{time.strftime('%Y')}, {__author__}."
__homepage__ = "https://github.com/Lightning-AI/pytorch-lightning"
__docs_url__ = "https://pytorch-lightning.readthedocs.io/en/stable/"
# this has to be simple string, see: https://github.com/pypa/twine/issues/522
__docs__ = (
    "PyTorch Lightning is the lightweight PyTorch wrapper for ML researchers."
    " Scale your models. Write less boilerplate."
)
__long_docs__ = """
Lightning is a way to organize your PyTorch code to decouple the science code from the engineering.
 It's more of a style-guide than a framework.

In Lightning, you organize your code into 3 distinct categories:

1. Research code (goes in the LightningModule).
2. Engineering code (you delete, and is handled by the Trainer).
3. Non-essential research code (logging, etc. this goes in Callbacks).

Although your research/production project might start simple, once you add things like GPU AND TPU training,
 16-bit precision, etc, you end up spending more time engineering than researching.
 Lightning automates AND rigorously tests those parts for you.

Overall, Lightning guarantees rigorously tested, correct, modern best practices for the automated parts.

Documentation
-------------
- https://pytorch-lightning.readthedocs.io/en/latest
- https://pytorch-lightning.readthedocs.io/en/stable
"""

__all__ = [
    "__author__",
    "__author_email__",
    "__copyright__",
    "__docs__",
    "__docs_url__",
    "__homepage__",
    "__license__",
]

_PACKAGE_ROOT = os.path.dirname(__file__)
_SOURCE_ROOT = os.path.dirname(_PACKAGE_ROOT)
_PROJECT_ROOT = os.path.dirname(_SOURCE_ROOT)
_PATH_REQUIREMENTS = os.path.join(_PROJECT_ROOT, "requirements", "pytorch")


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


_setup_tools = _load_py_module("setup_tools", os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py"))
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
        "examples": _setup_tools.load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="examples.txt"),
        "loggers": _setup_tools.load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="loggers.txt"),
        "extra": _setup_tools.load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="extra.txt"),
        "strategies": _setup_tools.load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="strategies.txt"),
        "test": _setup_tools.load_requirements(path_dir=_PATH_REQUIREMENTS, file_name="test.txt"),
    }
    for req in parse_requirements(extras["strategies"]):
        extras[req.key] = [str(req)]
    extras["dev"] = extras["extra"] + extras["loggers"] + extras["test"]
    extras["all"] = extras["dev"] + extras["examples"] + extras["strategies"]  # + extras['docs']
    return extras


def _setup_args():
    return dict(
        name="pytorch-lightning",
        version=__version.version,
        description=__docs__,
        author=__author__,
        author_email=__author_email__,
        url=__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=__license__,
        # todo: temp disable installing apps from source
        packages=find_packages(where="src", include=["pytorch_lightning", "pytorch_lightning.*"]),
        package_dir={"": "src"},
        include_package_data=True,
        long_description=_long_description,
        long_description_content_type="text/markdown",
        zip_safe=False,
        keywords=["deep learning", "pytorch", "AI"],
        python_requires=">=3.7",
        setup_requires=[],
        install_requires=_setup_tools.load_requirements(_PATH_REQUIREMENTS),
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


# fixme
print()
