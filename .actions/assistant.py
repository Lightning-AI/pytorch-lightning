# Copyright The Lightning AI team.
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
import logging
import os
import re
import shutil
import tempfile
import urllib.request
from collections.abc import Iterable, Iterator, Sequence
from itertools import chain
from os.path import dirname, isfile
from pathlib import Path
from typing import Any, Optional

from packaging.requirements import Requirement
from packaging.version import Version

REQUIREMENT_FILES = {
    "pytorch": (
        "requirements/pytorch/base.txt",
        "requirements/pytorch/extra.txt",
        "requirements/pytorch/strategies.txt",
        "requirements/pytorch/examples.txt",
    ),
    "fabric": (
        "requirements/fabric/base.txt",
        "requirements/fabric/strategies.txt",
    ),
    "data": ("requirements/data/data.txt",),
}
REQUIREMENT_FILES_ALL = list(chain(*REQUIREMENT_FILES.values()))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))


class _RequirementWithComment(Requirement):
    strict_string = "# strict"

    def __init__(self, *args: Any, comment: str = "", pip_argument: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.comment = comment
        assert pip_argument is None or pip_argument  # sanity check that it's not an empty str
        self.pip_argument = pip_argument
        self.strict = self.strict_string in comment.lower()

    def adjust(self, unfreeze: str) -> str:
        """Remove version restrictions unless they are strict.

        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# anything").adjust("none")
        'arrow<=1.2.2,>=1.2.0'
        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# strict").adjust("none")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# my name").adjust("all")
        'arrow>=1.2.0'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust("all")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow").adjust("all")
        'arrow'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# cool").adjust("major")
        'arrow<2.0,>=1.2.0'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust("major")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow>=1.2.0").adjust("major")
        'arrow>=1.2.0'
        >>> _RequirementWithComment("arrow").adjust("major")
        'arrow'

        """
        out = str(self)
        if self.strict:
            return f"{out}  {self.strict_string}"
        specs = [(spec.operator, spec.version) for spec in self.specifier]
        if unfreeze == "major":
            for operator, version in specs:
                if operator in ("<", "<="):
                    major = Version(version).major
                    # replace upper bound with major version increased by one
                    return out.replace(f"{operator}{version}", f"<{major + 1}.0")
        elif unfreeze == "all":
            for operator, version in specs:
                if operator in ("<", "<="):
                    # drop upper bound
                    return out.replace(f"{operator}{version},", "")
        elif unfreeze != "none":
            raise ValueError(f"Unexpected unfreeze: {unfreeze!r} value.")
        return out


def _parse_requirements(lines: Iterable[str]) -> Iterator[_RequirementWithComment]:
    """Adapted from `pkg_resources.parse_requirements` to include comments.

    >>> txt = ['# ignored', '', 'this # is an', '--piparg', 'example', 'foo # strict', 'thing', '-r different/file.txt']
    >>> [r.adjust('none') for r in _parse_requirements(txt)]
    ['this', 'example', 'foo  # strict', 'thing']

    """
    pip_argument = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Drop comments -- a hash without a space may be in a URL.
        if " #" in line:
            comment_pos = line.find(" #")
            line, comment = line[:comment_pos], line[comment_pos:]
        else:
            comment = ""
        # If there's a pip argument, save it
        if line.startswith("--"):
            pip_argument = line
            continue
        if line.startswith("-r "):
            # linked requirement files are unsupported
            continue
        yield _RequirementWithComment(line, comment=comment, pip_argument=pip_argument)
        pip_argument = None


def load_requirements(path_dir: str, file_name: str = "base.txt", unfreeze: str = "all") -> list[str]:
    """Loading requirements from a file.

    >>> path_req = os.path.join(_PROJECT_ROOT, "requirements")
    >>> load_requirements(path_req, "docs.txt", unfreeze="major")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sphinx<...]

    """
    assert unfreeze in {"none", "major", "all"}
    path = Path(path_dir) / file_name
    if not path.exists():
        logging.warning(f"Folder {path_dir} does not have any base requirements.")
        return []
    assert path.exists(), (path_dir, file_name, path)
    text = path.read_text().splitlines()
    return [req.adjust(unfreeze) for req in _parse_requirements(text)]


def load_readme_description(path_dir: str, homepage: str, version: str) -> str:
    """Load readme as decribtion.

    >>> load_readme_description(_PROJECT_ROOT, "", "")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    '...PyTorch Lightning is just organized PyTorch...'

    """
    path_readme = os.path.join(path_dir, "README.md")
    with open(path_readme, encoding="utf-8") as fo:
        text = fo.read()

    # drop images from readme
    text = text.replace(
        "![PT to PL](docs/source-pytorch/_static/images/general/pl_quick_start_full_compressed.gif)", ""
    )

    # https://github.com/Lightning-AI/lightning/raw/master/docs/source/_static/images/lightning_module/pt_to_pl.png
    github_source_url = os.path.join(homepage, "raw", version)
    # replace relative repository path to absolute link to the release
    #  do not replace all "docs" as in the readme we reger some other sources with particular path to docs
    text = text.replace(
        "docs/source-pytorch/_static/", f"{os.path.join(github_source_url, 'docs/source-app/_static/')}"
    )

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
    return re.sub(rf"{skip_begin}.+?{skip_end}", "<!--  -->", text, flags=re.IGNORECASE + re.DOTALL)

    # # https://github.com/Borda/pytorch-lightning/releases/download/1.1.0a6/codecov_badge.png
    # github_release_url = os.path.join(homepage, "releases", "download", version)
    # # download badge and replace url with local file
    # text = _parse_for_badge(text, github_release_url)


def distribute_version(src_folder: str, ver_file: str = "version.info") -> None:
    """Copy the global version to all packages."""
    ls_ver = glob.glob(os.path.join(src_folder, "*", "__version__.py"))
    ver_template = os.path.join(src_folder, ver_file)
    for fpath in ls_ver:
        fpath = os.path.join(os.path.dirname(fpath), ver_file)
        print("Distributing the version to", fpath)
        if os.path.isfile(fpath):
            os.remove(fpath)
        shutil.copy2(ver_template, fpath)


def _load_aggregate_requirements(req_dir: str = "requirements", freeze_requirements: bool = False) -> None:
    """Load all base requirements from all particular packages and prune duplicates.

    >>> _load_aggregate_requirements(os.path.join(_PROJECT_ROOT, "requirements"))

    """
    requires = [
        load_requirements(d, unfreeze="none" if freeze_requirements else "major")
        for d in glob.glob(os.path.join(req_dir, "*"))
        # skip empty folder (git artifacts), and resolving Will's special issue
        if os.path.isdir(d) and len(glob.glob(os.path.join(d, "*"))) > 0 and not os.path.basename(d).startswith("_")
    ]
    if not requires:
        return
    # TODO: add some smarter version aggregation per each package
    requires = sorted(set(chain(*requires)))
    with open(os.path.join(req_dir, "base.txt"), "w") as fp:
        fp.writelines([ln + os.linesep for ln in requires] + [os.linesep])


def _retrieve_files(directory: str, *ext: str) -> list[str]:
    all_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if not ext or any(os.path.split(fname)[1].lower().endswith(e) for e in ext):
                all_files.append(os.path.join(root, fname))

    return all_files


def _replace_imports(lines: list[str], mapping: list[tuple[str, str]], lightning_by: str = "") -> list[str]:
    """Replace imports of standalone package to lightning.

    >>> lns = [
    ...     '"lightning_app"',
    ...     "lightning_app",
    ...     "lightning_app/",
    ...     "delete_cloud_lightning_apps",
    ...     "from lightning_app import",
    ...     "lightning_apps = []",
    ...     "lightning_app and pytorch_lightning are ours",
    ...     "def _lightning_app():",
    ...     ":class:`~lightning_app.core.flow.LightningFlow`",
    ...     "http://pytorch_lightning.ai",
    ...     "from lightning import __version__",
    ...     "@lightning.ai"
    ... ]
    >>> mapping = [("lightning_app", "lightning.app"), ("pytorch_lightning", "lightning.pytorch")]
    >>> _replace_imports(lns, mapping, lightning_by="lightning_fabric")  # doctest: +NORMALIZE_WHITESPACE
    ['"lightning.app"', \
     'lightning.app', \
     'lightning_app/', \
     'delete_cloud_lightning_apps', \
     'from lightning.app import', \
     'lightning_apps = []', \
     'lightning.app and lightning.pytorch are ours', \
     'def _lightning_app():', \
     ':class:`~lightning.app.core.flow.LightningFlow`', \
     'http://pytorch_lightning.ai', \
     'from lightning_fabric import __version__', \
     '@lightning.ai']

    """
    out = lines[:]
    for source_import, target_import in mapping:
        for i, ln in enumerate(out):
            out[i] = re.sub(
                rf"([^_/@]|^){source_import}([^_\w/]|$)",
                rf"\1{target_import}\2",
                ln,
            )
            if lightning_by:  # in addition, replace base package
                out[i] = out[i].replace("from lightning import ", f"from {lightning_by} import ")
                out[i] = out[i].replace("import lightning ", f"import {lightning_by} ")
    return out


def copy_replace_imports(
    source_dir: str,
    source_imports: Sequence[str],
    target_imports: Sequence[str],
    target_dir: Optional[str] = None,
    lightning_by: str = "",
) -> None:
    """Copy package content with import adjustments."""
    print(f"Replacing imports: {locals()}")
    assert len(source_imports) == len(target_imports), (
        "source and target imports must have the same length, "
        f"source: {len(source_imports)}, target: {len(target_imports)}"
    )
    if target_dir is None:
        target_dir = source_dir

    ls = _retrieve_files(source_dir)
    for fp in ls:
        fp_new = fp.replace(source_dir, target_dir)
        _, ext = os.path.splitext(fp)
        if ext in (".png", ".jpg", ".ico"):
            os.makedirs(dirname(fp_new), exist_ok=True)
            if not isfile(fp_new):
                shutil.copy(fp, fp_new)
            continue
        if ext in (".pyc",):
            continue
        # Try to parse everything else
        with open(fp, encoding="utf-8") as fo:
            try:
                lines = fo.readlines()
            except UnicodeDecodeError:
                # a binary file, skip
                print(f"Skipped replacing imports for {fp}")
                continue
        lines = _replace_imports(lines, list(zip(source_imports, target_imports)), lightning_by=lightning_by)
        os.makedirs(os.path.dirname(fp_new), exist_ok=True)
        with open(fp_new, "w", encoding="utf-8") as fo:
            fo.writelines(lines)


def create_mirror_package(source_dir: str, package_mapping: dict[str, str]) -> None:
    """Create a mirror package with adjusted imports."""
    # replace imports and copy the code
    mapping = package_mapping.copy()
    mapping.pop("lightning", None)  # pop this key to avoid replacing `lightning` to `lightning.lightning`

    mapping = {f"lightning.{sp}": sl for sp, sl in mapping.items()}
    for pkg_from, pkg_to in mapping.items():
        source_imports, target_imports = zip(*mapping.items())
        copy_replace_imports(
            source_dir=os.path.join(source_dir, pkg_from.replace(".", os.sep)),
            # pytorch_lightning uses lightning_fabric, so we need to replace all imports for all directories
            source_imports=source_imports,
            target_imports=target_imports,
            target_dir=os.path.join(source_dir, pkg_to.replace(".", os.sep)),
            lightning_by=pkg_from,
        )


class AssistantCLI:
    @staticmethod
    def requirements_prune_pkgs(packages: Sequence[str], req_files: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
        """Remove some packages from given requirement files."""
        if isinstance(req_files, str):
            req_files = [req_files]
        for req in req_files:
            AssistantCLI._prune_packages(req, packages)

    @staticmethod
    def _prune_packages(req_file: str, packages: Sequence[str]) -> None:
        """Remove some packages from given requirement files."""
        path = Path(req_file)
        assert path.exists()
        text = path.read_text()
        lines = text.splitlines()
        final = []
        for line in lines:
            ln_ = line.strip()
            if not ln_ or ln_.startswith("#"):
                final.append(line)
                continue
            req = list(_parse_requirements([ln_]))[0]
            if req.name not in packages:
                final.append(line)
        print(final)
        path.write_text("\n".join(final) + "\n")

    @staticmethod
    def _replace_min(fname: str) -> None:
        with open(fname, encoding="utf-8") as fo:
            req = fo.read().replace(">=", "==")
        with open(fname, "w", encoding="utf-8") as fw:
            fw.write(req)

    @staticmethod
    def replace_oldest_ver(requirement_fnames: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
        """Replace the min package version by fixed one."""
        for fname in requirement_fnames:
            print(fname)
            AssistantCLI._replace_min(fname)

    @staticmethod
    def copy_replace_imports(
        source_dir: str,
        source_import: str,
        target_import: str,
        target_dir: Optional[str] = None,
        lightning_by: str = "",
    ) -> None:
        """Copy package content with import adjustments."""
        source_imports = source_import.strip().split(",")
        target_imports = target_import.strip().split(",")
        copy_replace_imports(
            source_dir, source_imports, target_imports, target_dir=target_dir, lightning_by=lightning_by
        )

    @staticmethod
    def pull_docs_files(
        gh_user_repo: str,
        target_dir: str = "docs/source-pytorch/XXX",
        checkout: str = "refs/tags/1.0.0",
        source_dir: str = "docs/source",
        single_page: Optional[str] = None,
        as_orphan: bool = False,
    ) -> None:
        """Pull docs pages from external source and append to local docs.

        Args:
            gh_user_repo: standard GitHub user/repo string
            target_dir: relative location inside the docs folder
            checkout: specific tag or branch to checkout
            source_dir: relative location inside the remote / external repo
            single_page: copy only single page from the remote repo and name it as the repo name
            as_orphan: append orphan statement to the page

        """
        import zipfile

        zip_url = f"https://github.com/{gh_user_repo}/archive/{checkout}.zip"

        with tempfile.TemporaryDirectory() as tmp:
            zip_file = os.path.join(tmp, "repo.zip")
            try:
                urllib.request.urlretrieve(zip_url, zip_file)
            except urllib.error.HTTPError:
                raise RuntimeError(f"Requesting file '{zip_url}' does not exist or it is just unavailable.")

            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(tmp)

            zip_dirs = [d for d in glob.glob(os.path.join(tmp, "*")) if os.path.isdir(d)]
            # check that the extracted archive has only repo folder
            assert len(zip_dirs) == 1
            repo_dir = zip_dirs[0]

            if single_page:  # special case for copying single page
                single_page = os.path.join(repo_dir, source_dir, single_page)
                assert os.path.isfile(single_page), f"File '{single_page}' does not exist."
                name = re.sub(r"lightning[-_]?", "", gh_user_repo.split("/")[-1])
                new_rst = os.path.join(_PROJECT_ROOT, target_dir, f"{name}.rst")
                AssistantCLI._copy_rst(single_page, new_rst, as_orphan=as_orphan)
                return
            # continue with copying all pages
            ls_pages = glob.glob(os.path.join(repo_dir, source_dir, "*.rst"))
            ls_pages += glob.glob(os.path.join(repo_dir, source_dir, "**", "*.rst"))
            for rst in ls_pages:
                rel_rst = rst.replace(os.path.join(repo_dir, source_dir) + os.path.sep, "")
                rel_dir = os.path.dirname(rel_rst)
                os.makedirs(os.path.join(_PROJECT_ROOT, target_dir, rel_dir), exist_ok=True)
                new_rst = os.path.join(_PROJECT_ROOT, target_dir, rel_rst)
                if os.path.isfile(new_rst):
                    logging.warning(f"Page {new_rst} already exists in the local tree so it will be skipped.")
                    continue
                AssistantCLI._copy_rst(rst, new_rst, as_orphan=as_orphan)

    @staticmethod
    def _copy_rst(rst_in, rst_out, as_orphan: bool = False):
        """Copy RST page with optional inserting orphan statement."""
        with open(rst_in, encoding="utf-8") as fopen:
            page = fopen.read()
        if as_orphan and ":orphan:" not in page:
            page = ":orphan:\n\n" + page
        with open(rst_out, "w", encoding="utf-8") as fopen:
            fopen.write(page)

    @staticmethod
    def convert_version2nightly(ver_file: str = "src/version.info") -> None:
        """Load the actual version and convert it to the nightly version."""
        from datetime import datetime

        with open(ver_file) as fo:
            version = fo.read().strip()
        # parse X.Y.Z version and prune any suffix
        vers = re.match(r"(\d+)\.(\d+)\.(\d+).*", version)
        # create timestamp  YYYYMMDD
        timestamp = datetime.now().strftime("%Y%m%d")
        version = f"{'.'.join(vers.groups())}.dev{timestamp}"
        with open(ver_file, "w") as fo:
            fo.write(version + os.linesep)

    @staticmethod
    def generate_docker_tags(
        release_version: str,
        python_version: str,
        torch_version: str,
        cuda_version: str,
        docker_project: str = "pytorchlightning/pytorch_lightning",
        add_latest: bool = False,
    ) -> None:
        """Generate docker tags for the given versions."""
        tags = [f"latest-py{python_version}-torch{torch_version}-cuda{cuda_version}"]
        if release_version:
            tags += [f"{release_version}-py{python_version}-torch{torch_version}-cuda{cuda_version}"]
        if add_latest:
            tags += ["latest"]

        tags = [f"{docker_project}:{tag}" for tag in tags]
        print(",".join(tags))


if __name__ == "__main__":
    import jsonargparse

    jsonargparse.CLI(AssistantCLI, as_positional=False)
