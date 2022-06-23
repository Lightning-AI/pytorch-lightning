import os
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType

from setuptools import find_packages

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "lightning_app")
_PATH_REQUIREMENTS = os.path.join("requirements", "app")


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


_path_setup_tools = os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py")
assert os.path.isfile(_path_setup_tools)
_setup_tools = _load_py_module("setup_tools", _path_setup_tools)
_about = _load_py_module("about", os.path.join(_PACKAGE_ROOT, "__about__.py"))
_version = _load_py_module("version", os.path.join(_PACKAGE_ROOT, "__version__.py"))
_long_description = _setup_tools.load_readme_description(
    _PACKAGE_ROOT, homepage=_about.__homepage__, version=_version.version
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
        version=_version.version,  # todo: consider using date version + branch for installation from source
        description=_about.__docs__,
        author=_about.__author__,
        author_email=_about.__author_email__,
        url=_about.__homepage__,
        download_url="https://github.com/Lightning-AI/lightning",
        license=_about.__license__,
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
