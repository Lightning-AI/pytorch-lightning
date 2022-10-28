# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import os
import pathlib
import re
import shutil
import tarfile
import tempfile
import urllib.request
from distutils.version import LooseVersion
from importlib.util import module_from_spec, spec_from_file_location
from itertools import chain
from types import ModuleType
from typing import Dict, List

from pkg_resources import parse_requirements

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
_PACKAGE_MAPPING = {"pytorch": "pytorch_lightning", "app": "lightning_app", "lite": "lightning_lite"}

# TODO: remove this once lightning-ui package is ready as a dependency
_LIGHTNING_FRONTEND_RELEASE_URL = "https://storage.googleapis.com/grid-packages/lightning-ui/v0.0.0/build.tar.gz"


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    assert spec, f"Failed to load module {name} from {location}"
    py = module_from_spec(spec)
    assert spec.loader, f"ModuleSpec.loader is None for {name} from {location}"
    spec.loader.exec_module(py)
    return py


def _augment_requirement(ln: str, comment_char: str = "#", unfreeze: str = "all") -> str:
    """Adjust the upper version contrains.

    Args:
        ln: raw line from requirement
        comment_char: charter marking comment
        unfreeze: Enum or "all"|"major"|""

    Returns:
        adjusted requirement

    >>> _augment_requirement("arrow>=1.2.0, <=1.2.2  # anything", unfreeze="")
    'arrow>=1.2.0, <=1.2.2'
    >>> _augment_requirement("arrow>=1.2.0, <=1.2.2  # strict", unfreeze="")
    'arrow>=1.2.0, <=1.2.2  # strict'
    >>> _augment_requirement("arrow>=1.2.0, <=1.2.2  # my name", unfreeze="all")
    'arrow>=1.2.0'
    >>> _augment_requirement("arrow>=1.2.0, <=1.2.2  # strict", unfreeze="all")
    'arrow>=1.2.0, <=1.2.2  # strict'
    >>> _augment_requirement("arrow", unfreeze="all")
    'arrow'
    >>> _augment_requirement("arrow>=1.2.0, <=1.2.2  # cool", unfreeze="major")
    'arrow>=1.2.0, <2.0  # strict'
    >>> _augment_requirement("arrow>=1.2.0, <=1.2.2  # strict", unfreeze="major")
    'arrow>=1.2.0, <=1.2.2  # strict'
    >>> _augment_requirement("arrow>=1.2.0", unfreeze="major")
    'arrow>=1.2.0, <2.0  # strict'
    >>> _augment_requirement("arrow", unfreeze="major")
    'arrow'
    """
    # filer all comments
    if comment_char in ln:
        comment = ln[ln.index(comment_char) :]
        ln = ln[: ln.index(comment_char)]
        is_strict = "strict" in comment
    else:
        is_strict = False
    req = ln.strip()
    # skip directly installed dependencies
    if not req or req.startswith("http") or "@" in req:
        return ""
    # extract the major version from all listed versions
    if unfreeze == "major":
        req_ = list(parse_requirements([req]))[0]
        vers = [LooseVersion(v) for s, v in req_.specs if s not in ("==", "~=")]
        ver_major = sorted(vers)[-1].version[0] if vers else None
    else:
        ver_major = None

    # remove version restrictions unless they are strict
    if unfreeze and "<" in req and not is_strict:
        req = re.sub(r",? *<=? *[\d\.\*]+", "", req).strip()
    if ver_major is not None and not is_strict:
        # add , only if there are already some versions
        req += f"{',' if any(c in req for c in '<=>') else ''} <{int(ver_major) + 1}.0"

    # adding strict back to the comment
    if is_strict or ver_major is not None:
        req += "  # strict"

    return req


def load_requirements(
    path_dir: str, file_name: str = "base.txt", comment_char: str = "#", unfreeze: str = "all"
) -> List[str]:
    """Loading requirements from a file.

    >>> path_req = os.path.join(_PROJECT_ROOT, "requirements")
    >>> load_requirements(path_req, "docs.txt", unfreeze="major")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sphinx>=4.0, <6.0  # strict', ...]
    """
    with open(os.path.join(path_dir, file_name)) as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = [_augment_requirement(ln, comment_char=comment_char, unfreeze=unfreeze) for ln in lines]
    # filter empty lines and containing @ which means redirect to some git/http
    reqs = [str(req) for req in reqs if req and not any(c in req for c in ["@", "http:", "https:"])]
    return reqs


def load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as decribtion.

    >>> load_readme_description(_PROJECT_ROOT, "", "")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '...'
    """
    path_readme = os.path.join(path_dir, "README.md")
    text = open(path_readme, encoding="utf-8").read()

    # drop images from readme
    text = text.replace("![PT to PL](docs/source/_static/images/general/pl_quick_start_full_compressed.gif)", "")

    # https://github.com/Lightning-AI/lightning/raw/master/docs/source/_static/images/lightning_module/pt_to_pl.png
    github_source_url = os.path.join(homepage, "raw", version)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we reger some other sources with particular path to docs
    text = text.replace("docs/source/_static/", f"{os.path.join(github_source_url, 'docs/source/_static/')}")

    # readthedocs badge
    text = text.replace("badge/?version=stable", f"badge/?version={version}")
    text = text.replace("pytorch-lightning.readthedocs.io/en/stable/", f"pytorch-lightning.readthedocs.io/en/{version}")
    # codecov badge
    text = text.replace("/branch/master/graph/badge.svg", f"/release/{version}/graph/badge.svg")
    # github actions badge
    text = text.replace("badge.svg?branch=master&event=push", f"badge.svg?tag={version}")
    # azure pipelines badge
    text = text.replace("?branchName=master", f"?branchName=refs%2Ftags%2F{version}")

    skip_begin = r"<!-- following section will be skipped from PyPI description -->"
    skip_end = r"<!-- end skipping PyPI description -->"
    # todo: wrap content as commented description
    text = re.sub(rf"{skip_begin}.+?{skip_end}", "<!--  -->", text, flags=re.IGNORECASE + re.DOTALL)

    # # https://github.com/Borda/pytorch-lightning/releases/download/1.1.0a6/codecov_badge.png
    # github_release_url = os.path.join(homepage, "releases", "download", version)
    # # download badge and replace url with local file
    # text = _parse_for_badge(text, github_release_url)
    return text


def parse_version_from_file(pkg_root: str) -> str:
    """Loading the package version from file."""
    file_ver = os.path.join(pkg_root, "__version__.py")
    file_about = os.path.join(pkg_root, "__about__.py")
    if os.path.isfile(file_ver):
        ver = _load_py_module("version", file_ver).version
    elif os.path.isfile(file_about):
        ver = _load_py_module("about", file_about).__version__
    else:  # this covers case you have build only meta-package so not additional source files are present
        ver = ""
    return ver


def _replace_imports_in_file(lines: List[str], pkg_lut: Dict[str, str]) -> List[str]:
    """Replace imports of standalone package to lightning.

    >>> lns = ["lightning_app",
    ...        "delete_cloud_lightning_apps",
    ...        "from lightning_app import",
    ...        "lightning_apps = []",
    ...        "lightning_app is ours",
    ...        "def _lightning_app():",
    ...        ":class:`~lightning_app.core.flow.LightningFlow`"]
    >>> from pprint import pprint
    >>> pprint(_replace_imports_in_file(lns, {"app": "lightning_app"}))
    ['lightning.app',
     'delete_cloud_lightning_apps',
     'from lightning.app import',
     'lightning_apps = []',
     'lightning.app is ours',
     'def _lightning_app():',
     ':class:`~lightning.app.core.flow.LightningFlow`']
    """
    for n2, n1 in pkg_lut.items():
        for i, ln in enumerate(lines):
            lines[i] = re.sub(rf"([^_]|^){n1}([^_\w]|$)", rf"\1lightning.{n2}\2", ln)
    return lines


# TODO: unify usage with assistant function, such that import this function in there
def copy_adjusted_modules(src_folder: str, pkg_name: str, lit_name: str, pkg_lut: dict) -> None:
    """Recursively replace imports in given folder."""
    package_dir = os.path.join(src_folder, pkg_name)
    all_files = glob.glob(os.path.join(package_dir, "**", "*.*"), recursive=True)
    for fname in all_files:
        local_path = fname.replace(package_dir + os.path.sep, "")
        new_file = os.path.join(src_folder, "lightning", lit_name, local_path)
        if not fname.endswith(".py"):
            if not fname.endswith(".pyc"):
                os.makedirs(os.path.dirname(new_file), exist_ok=True)
                shutil.copy2(fname, new_file)
            continue

        with open(fname, encoding="utf-8") as fo:
            py = fo.readlines()
        py = _replace_imports_in_file(py, pkg_lut)
        os.makedirs(os.path.dirname(new_file), exist_ok=True)
        with open(new_file, "w", encoding="utf-8") as fo:
            fo.writelines(py)


def create_mirror_package(src_folder: str, lit_pkg_mapping: dict) -> None:
    """Recursively replace imports in given folder.

    >>> create_mirror_package(
    ...     os.path.join(_PROJECT_ROOT, "src"),
    ...     {"pytorch": "pytorch_lightning", "app": "lightning_app", "lite": "lightning_lite"}
    ... )
    """
    for lit_name, pkg_name in lit_pkg_mapping.items():
        copy_adjusted_modules(src_folder, pkg_name, lit_name, lit_pkg_mapping)


def _download_frontend(pkg_path: str):
    """Downloads an archive file for a specific release of the Lightning frontend and extracts it to the correct
    directory."""

    try:
        frontend_dir = pathlib.Path(pkg_path, "ui")
        download_dir = tempfile.mkdtemp()

        shutil.rmtree(frontend_dir, ignore_errors=True)
        response = urllib.request.urlopen(_LIGHTNING_FRONTEND_RELEASE_URL)

        file = tarfile.open(fileobj=response, mode="r|gz")
        file.extractall(path=download_dir)

        shutil.move(os.path.join(download_dir, "build"), frontend_dir)
        print("The Lightning UI has successfully been downloaded!")

    # If installing from source without internet connection, we don't want to break the installation
    except Exception:
        print("The Lightning UI downloading has failed!")


def _load_aggregate_requirements(req_dir: str = "requirements", freeze_requirements: bool = False) -> None:
    """Load all base requirements from all particular packages and prune duplicates."""
    requires = [
        load_requirements(d, file_name="base.txt", unfreeze=not freeze_requirements)
        for d in glob.glob(os.path.join(req_dir, "*"))
        # skip empty folder as git artefacts, and resolving Will's special issue
        if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*"))) > 0
    ]
    if not requires:
        return None
    # TODO: add some smarter version aggregation per each package
    requires = list(chain(*requires))
    with open(os.path.join(req_dir, "base.txt"), "w") as fp:
        fp.writelines([ln + os.linesep for ln in requires])


def set_actual_version_from_src(req_path: str, src_root: str, pkg_name: str) -> None:
    """Setting actual version from source code for a given package."""
    with open(req_path, encoding="utf-8") as fo:
        lines = fo.readlines()
    ver = parse_version_from_file(os.path.join(src_root, pkg_name.replace("-", "_")))
    for i, ln in enumerate(lines):
        reqs = list(parse_requirements([ln]))
        if not reqs:
            continue
        if reqs[0].name == pkg_name:
            lines[i] = f"{pkg_name}=={ver}{os.linesep}"

    with open(req_path, "w", encoding="utf-8") as fw:
        fw.writelines(lines)
