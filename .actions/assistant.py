import datetime
import json
import os
import re
from pprint import pprint
from typing import Dict, Sequence
from urllib import request

import fire
import yaml
from pkg_resources import parse_version

REQUIREMENT_FILES = (
    "requirements.txt",
    "requirements/extra.txt",
    "requirements/loggers.txt",
    # "requirements/test.txt",
    "requirements/examples.txt",
)
REQUIREMENT_LOCK = "requirements/locked.yaml"


class Requirements:
    """Manipulation with requirements file."""

    def __init__(self, fname: str, show_final: bool = False):
        self._fname = fname
        with open(self._fname) as fp:
            self.lines = fp.readlines()
        self.lines = [ln.strip() for ln in self.lines]
        self._show_final = show_final

    def __enter__(self):
        return self

    def __exit__(self, type_, value_, traceback_):
        if self._show_final:
            print(os.linesep.join(self.lines))
        self.lines = [f"{ln}{os.linesep}" for ln in self.lines]
        with open(self._fname, "w") as fp:
            fp.writelines(self.lines)


def versions(pkg_name):
    url = f"https://pypi.python.org/pypi/{pkg_name}/json"
    releases = json.loads(request.urlopen(url).read())["releases"]
    return sorted(releases, key=parse_version, reverse=True)


class AssistantCLI:

    _PATH_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

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
    def requirements_prune_pkgs(req_file: str, packages: Sequence[str]) -> None:
        """Remove some packages from given requirement files."""
        if isinstance(packages, str):
            packages = [packages]
        with Requirements(req_file, show_final=True) as req:
            for pkg in packages:
                req.lines = [ln for ln in req.lines if not ln.startswith(pkg)]

    @staticmethod
    def replace_oldest_versions(req_files: Sequence[str] = REQUIREMENT_FILES) -> None:
        """Replace the min package version by fixed one."""
        for fname in req_files:
            req = open(fname).read().replace(">=", "==")
            open(fname, "w").write(req)

    @staticmethod
    def _lock_pkg_version(req: str, locked: Dict[str, str], comment_char: str = "#") -> str:
        """For each line/package find locked version from specific file."""
        if comment_char in req:
            req = req[: req.index(comment_char)].strip()
        if not req:  # if requirement is not empty
            return req
        sep_idx = [req.index(c) for c in "<=>[]" if c in req]
        name = (req[: min(sep_idx)] if sep_idx else req).strip()
        if name not in locked:
            return req
        sep_needed = any(c in req for c in "<=>")
        req += f"{',' if sep_needed else ''} <={locked[name]}"
        return req

    @staticmethod
    def replace_locked_versions(
        req_files: Sequence[str] = REQUIREMENT_FILES, lock_file: str = REQUIREMENT_LOCK
    ) -> None:
        """Replace the package version by locked one."""
        with open(lock_file) as fp:
            locked = yaml.safe_load(fp)
        if isinstance(req_files, str):
            req_files = [req_files]

        for fname in req_files:
            with Requirements(fname, show_final=True) as req:
                req.lines = [AssistantCLI._lock_pkg_version(ln, locked) for ln in req.lines]

    @staticmethod
    def _latest_pkg_version(req: str, comment_char: str = "#") -> tuple:
        """Ask PyPI about the latest available package version on the stack."""
        sep_idx = [req.index(c) for c in "<=>[]" + comment_char if c in req]
        name = (req[: min(sep_idx)] if sep_idx else req).strip()
        if not name:
            return None, None
        ver = versions(name)[0]
        return name, ver

    @staticmethod
    def create_locked_versions(req_files: Sequence[str] = REQUIREMENT_FILES, lock_file: str = REQUIREMENT_LOCK) -> None:
        """Create the min package version by fixed one."""
        if isinstance(req_files, str):
            req_files = [req_files]

        locked = {}
        for fname in req_files:
            with Requirements(fname) as req:
                pkg_versions = [AssistantCLI._latest_pkg_version(ln) for ln in req.lines]
                locked.update({pkg: str(ver) for pkg, ver in pkg_versions if pkg})

        pprint(locked)
        with open(lock_file, "w") as fp:
            yaml.safe_dump(locked, fp)


if __name__ == "__main__":
    fire.Fire(AssistantCLI)
