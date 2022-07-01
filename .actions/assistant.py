import datetime
import json
import os
import re
from distutils.version import LooseVersion, StrictVersion
from importlib.util import module_from_spec, spec_from_file_location
from itertools import chain
from pathlib import Path
from pprint import pprint
from types import ModuleType
from typing import List, Optional, Sequence
from urllib.request import Request, urlopen

import fire

REQUIREMENT_FILES = {
    "pytorch": (
        "requirements/pytorch/base.txt",
        "requirements/pytorch/extra.txt",
        "requirements/pytorch/loggers.txt",
        "requirements/pytorch/strategies.txt",
        "requirements/pytorch/examples.txt",
    )
}
REQUIREMENT_FILES_ALL = tuple(chain(*REQUIREMENT_FILES.values()))
PACKAGE_MAPPING = {"app": "lightning-app", "pytorch": "pytorch-lightning"}


def pypi_versions(package_name: str) -> List[str]:
    # https://stackoverflow.com/a/27239645/4521646
    url = f"https://pypi.org/pypi/{package_name}/json"
    data = json.load(urlopen(Request(url)))
    versions = list(data["releases"].keys())
    # todo: drop this line after cleaning Pypi history from invalid versions
    versions = list(filter(lambda v: v.count(".") == 2 and "rc" not in v, versions))
    versions.sort(key=StrictVersion)
    return versions


def _load_py_module(name: str, location: str) -> ModuleType:
    spec = spec_from_file_location(name, location)
    py = module_from_spec(spec)
    spec.loader.exec_module(py)
    return py


class AssistantCLI:
    _PATH_ROOT = str(Path(__file__).parent.parent)
    _PATH_SRC = os.path.join(_PATH_ROOT, "src")

    @staticmethod
    def prepare_nightly_version(proj_root: str = _PATH_ROOT) -> None:
        """Replace semantic version by date."""
        path_info = os.path.join(proj_root, "pytorch_lightning", "__about__.py")
        # get today date
        now = datetime.datetime.now()
        now_date = now.strftime("%Y%m%d")

        print(f"prepare init '{path_info}' - replace version by {now_date}")
        with open(path_info) as fp:
            init = fp.read()
        init = re.sub(r'__version__ = [\d\.\w\'"]+', f'__version__ = "{now_date}"', init)
        with open(path_info, "w") as fp:
            fp.write(init)

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
        with open(req_file) as fp:
            lines = fp.readlines()

        if isinstance(packages, str):
            packages = [packages]
        for pkg in packages:
            lines = [ln for ln in lines if not ln.startswith(pkg)]
        pprint(lines)

        with open(req_file, "w") as fp:
            fp.writelines(lines)

    @staticmethod
    def _replace_min(fname: str) -> None:
        req = open(fname).read().replace(">=", "==")
        open(fname, "w").write(req)

    @staticmethod
    def replace_oldest_ver(requirement_fnames: Sequence[str] = REQUIREMENT_FILES_ALL) -> None:
        """Replace the min package version by fixed one."""
        for fname in requirement_fnames:
            AssistantCLI._replace_min(fname)

    @staticmethod
    def _release_pkg(pkg: str, src_folder: str = _PATH_SRC) -> bool:
        pypi_ver = pypi_versions(pkg)[-1]
        _version = _load_py_module("version", os.path.join(src_folder, pkg.replace("-", "_"), "__version__.py"))
        local_ver = _version.version
        return "dev" not in local_ver and LooseVersion(local_ver) > LooseVersion(pypi_ver)

    @staticmethod
    def determine_releasing_pkgs(
        src_folder: str = _PATH_SRC, packages: Sequence[str] = ("pytorch-lightning", "lightning-app")
    ) -> Optional[Sequence[str]]:
        if isinstance(packages, str):
            packages = [packages]
        releasing = [pkg for pkg in packages if AssistantCLI._release_pkg(PACKAGE_MAPPING[pkg], src_folder=src_folder)]
        return releasing


if __name__ == "__main__":
    fire.Fire(AssistantCLI)
