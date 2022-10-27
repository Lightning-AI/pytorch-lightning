# in case you have not installed the package and have only set python path to this source

import subprocess


def get_git_revision_short_hash() -> str:
    # https://stackoverflow.com/a/21901260/4521646
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()


try:
    version: str = get_git_revision_short_hash()
except Exception:
    version: str = ""
