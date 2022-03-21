import datetime
import os
import re
from typing import Dict, Sequence

import fire
import yaml

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
        if comment_char in req:
            req = req[: req.index(comment_char)].strip()
        if not req:  # if requirement is not empty
            return req
        sep_idx = [req.index(c) for c in "<=>" if c in req]
        name = (req[: min(sep_idx)] if sep_idx else req).strip()
        if name not in locked:
            return req
        return req + f", <={locked[name]}"

    @staticmethod
    def replace_locked_versions(
        req_files: Sequence[str] = REQUIREMENT_FILES, lock_file: str = REQUIREMENT_LOCK
    ) -> None:
        """Replace the min package version by fixed one."""
        with open(lock_file) as fp:
            locked = yaml.safe_load(fp)
        if isinstance(req_files, str):
            req_files = [req_files]

        for fname in req_files:
            with Requirements(fname, show_final=True) as req:
                req.lines = [AssistantCLI._lock_pkg_version(ln, locked) for ln in req.lines]


if __name__ == "__main__":
    fire.Fire(AssistantCLI)
