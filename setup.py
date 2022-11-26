#!/usr/bin/env python
import glob
import os
from importlib.util import module_from_spec, spec_from_file_location

from pkg_resources import parse_requirements
from setuptools import find_packages, setup

_PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
_PATH_SOURCE = os.path.join(_PATH_ROOT, "src")
_PATH_REQUIRE = os.path.join(_PATH_ROOT, "requirements")


def _load_py_module(fname, pkg="lightning_utilities"):
    spec = spec_from_file_location(os.path.join(pkg, fname), os.path.join(_PATH_SOURCE, pkg, fname))
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


about = _load_py_module("__about__.py")

# load basic requirements
with open(os.path.join(_PATH_REQUIRE, "base.txt")) as fp:
    requirements = list(map(str, parse_requirements(fp.readlines())))

# make extras as automated loading
requirements_extra = {}
for fpath in glob.glob(os.path.join(_PATH_REQUIRE, "*.txt")):
    if os.path.basename(fpath) == "base.txt":
        continue
    name, _ = os.path.splitext(os.path.basename(fpath))
    with open(fpath) as fp:
        requirements_extra[name] = list(map(str, parse_requirements(fp.readline())))

# loading readme as description
with open(os.path.join(_PATH_ROOT, "README.md")) as fp:
    readme = fp.read()

setup(
    name="lightning-utilities",
    version=about.__version__,
    description=about.__docs__,
    author=about.__author__,
    author_email=about.__author_email__,
    url=about.__homepage__,
    download_url="https://github.com/Lightning-AI/utilities",
    license=about.__license__,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    zip_safe=False,
    keywords=["Utilities", "DevOps", "CI/CD"],
    python_requires=">=3.7",
    setup_requires=[],
    install_requires=requirements,
    extras_require=requirements_extra,
    project_urls={
        "Bug Tracker": "https://github.com/Lightning-AI/utilities/issues",
        "Documentation": "https://dev-toolbox.rtfd.io/en/latest/",  # TODO: Update domain
        "Source Code": "https://github.com/Lightning-AI/utilities",
    },
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        # How mature is this project? Common values are
        #   3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        "Intended Audience :: Developers",
        # Pick your license as you wish
        # 'License :: OSI Approved :: BSD License',
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
