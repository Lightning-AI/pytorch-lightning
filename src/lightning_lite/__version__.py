import os

_PACKAGE_ROOT = os.path.dirname(__file__)
_SDIST_PATH = _VERSION_PATH = os.path.join(os.path.dirname(_PACKAGE_ROOT), "version.info")
if not os.path.exists(_SDIST_PATH):
    # relevant for `bdist_wheel`
    _VERSION_PATH = os.path.join(_PACKAGE_ROOT, "version.info")
with open(_VERSION_PATH, encoding="utf-8") as fo:
    version = fo.readlines()[0].strip()
