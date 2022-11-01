# in case you have not installed the package and have only set python path to this source

import os

_PATH_PKG = os.path.dirname(__file__)
_PATH_VER = os.path.join(_PATH_PKG, "version.info")

if os.path.isfile(_PATH_VER):
    with open(_PATH_VER, encoding="utf-8") as fo:
        version = fo.readlines()[0].strip()
else:
    try:
        import subprocess

        version: str = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("ascii").strip()
    except Exception:
        version = ""
