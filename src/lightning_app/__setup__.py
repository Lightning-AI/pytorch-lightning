import os
from importlib.util import module_from_spec, spec_from_file_location
from types import ModuleType
from typing import Any, Dict

from setuptools import find_packages

_PROJECT_ROOT = "."
_SOURCE_ROOT = os.path.join(_PROJECT_ROOT, "src")
_PACKAGE_ROOT = os.path.join(_SOURCE_ROOT, "lightning_app")
_PATH_REQUIREMENTS = os.path.join("requirements", "app")
_FREEZE_REQUIREMENTS = bool(int(os.environ.get("FREEZE_REQUIREMENTS", 0)))


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


def _prepare_extras() -> Dict[str, Any]:
    path_setup_tools = os.path.join(_PROJECT_ROOT, ".actions", "setup_tools.py")
    setup_tools = _load_py_module("setup_tools", path_setup_tools)
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras
    # Define package extras. These are only installed if you specify them.
    # From remote, use like `pip install pytorch-lightning[dev, docs]`
    # From local copy of repo, use like `pip install ".[dev, docs]"`
    common_args = dict(path_dir=_PATH_REQUIREMENTS, unfreeze="major" if _FREEZE_REQUIREMENTS else "all")
    extras = {
        # 'docs': load_requirements(file_name='docs.txt'),
        "cloud": setup_tools.load_requirements(file_name="cloud.txt", **common_args),
        "ui": setup_tools.load_requirements(file_name="ui.txt", **common_args),
        "test": setup_tools.load_requirements(file_name="test.txt", **common_args),
    }
    extras["extra"] = extras["cloud"] + extras["ui"]
    extras["dev"] = extras["extra"] + extras["test"]  # + extras['docs']
    extras["all"] = extras["dev"]
    return extras


def _adjust_manifest(**__: Any) -> None:
    manifest_path = os.path.join(_PROJECT_ROOT, "MANIFEST.in")
    assert os.path.isfile(manifest_path)
    with open(manifest_path) as fp:
        lines = fp.readlines()
    lines += [
        "recursive-exclude src *.md" + os.linesep,
        "recursive-exclude requirements *.txt" + os.linesep,
        "recursive-include src/lightning_app *.md" + os.linesep,
        "include src/lightning_app/components/serve/catimage.png" + os.linesep,
        "recursive-include requirements/app *.txt" + os.linesep,
        "recursive-include src/lightning_app/cli/*-template *" + os.linesep,  # Add templates
    ]

    # TODO: remove this once lightning-ui package is ready as a dependency
    lines += ["recursive-include src/lightning_app/ui *" + os.linesep]

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

    # TODO: remove this once lightning-ui package is ready as a dependency
    _setup_tools._download_frontend(_PACKAGE_ROOT)

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
        install_requires=_setup_tools.load_requirements(
            _PATH_REQUIREMENTS, unfreeze="major" if _FREEZE_REQUIREMENTS else "all"
        ),
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
