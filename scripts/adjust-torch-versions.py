# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Adjusting version across PTorch ecosystem."""
import logging
import os
import re
import sys
from typing import Dict, Optional

from packaging.version import Version

VERSIONS = [
    {"torch": "2.0.1", "torchvision": "0.15.2", "torchtext": "0.15.2"},  # stable
    {"torch": "2.0.0", "torchvision": "0.15.1", "torchtext": "0.15.1"},
    {"torch": "1.14.0", "torchvision": "0.15.0", "torchtext": "0.15.0"},  # nightly
    {"torch": "1.13.1", "torchvision": "0.14.1", "torchtext": "0.14.1"},  # stable
    {"torch": "1.13.0", "torchvision": "0.14.0", "torchtext": "0.14.0"},
    {"torch": "1.12.1", "torchvision": "0.13.1", "torchtext": "0.13.1"},
    {"torch": "1.12.0", "torchvision": "0.13.0", "torchtext": "0.13.0"},
    {"torch": "1.11.0", "torchvision": "0.12.0", "torchtext": "0.12.0"},
    {"torch": "1.10.2", "torchvision": "0.11.3", "torchtext": "0.11.2"},
    {"torch": "1.10.1", "torchvision": "0.11.2", "torchtext": "0.11.1"},
    {"torch": "1.10.0", "torchvision": "0.11.1", "torchtext": "0.11.0"},
    {"torch": "1.9.1", "torchvision": "0.10.1", "torchtext": "0.10.1"},
    {"torch": "1.9.0", "torchvision": "0.10.0", "torchtext": "0.10.0"},
    {"torch": "1.8.2", "torchvision": "0.9.1", "torchtext": "0.9.1"},
    {"torch": "1.8.1", "torchvision": "0.9.1", "torchtext": "0.9.1"},
    {"torch": "1.8.0", "torchvision": "0.9.0", "torchtext": "0.9.0"},
]
VERSIONS.sort(key=lambda v: Version(v["torch"]), reverse=True)


def find_latest(ver: str) -> Dict[str, str]:
    """Find the latest version."""
    # drop all except semantic version
    ver = re.search(r"([\.\d]+)", ver).groups()[0]
    # in case there remaining dot at the end - e.g "1.9.0.dev20210504"
    ver = ver[:-1] if ver[-1] == "." else ver
    logging.info(f"finding ecosystem versions for: {ver}")

    # find first match
    for option in VERSIONS:
        if option["torch"].startswith(ver):
            return option

    raise ValueError(f"Missing {ver} in {VERSIONS}")


def adjust(requires: str, torch_version: Optional[str] = None) -> str:
    """Adjust the versions to be paired within pytorch ecosystem."""
    if not torch_version:
        import torch

        torch_version = torch.__version__
    if not torch_version:
        raise ValueError(f"invalid torch: {torch_version}")

    # remove comments and strip whitespace
    requires = re.sub(rf"\s*#.*{os.linesep}", os.linesep, requires).strip()

    latest = find_latest(torch_version)
    for lib, version in latest.items():
        replace = f"{lib}=={version}" if version else ""
        requires = re.sub(rf"\b{lib}(?![-_\w]).*", replace, requires)

    return requires


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) == 3:
        requirements_path, torch_version = sys.argv[1:]
    else:
        requirements_path, torch_version = sys.argv[1], None
    logging.info(f"requirements_path='{requirements_path}' with arg torch_version='{torch_version}'")

    with open(requirements_path) as fp:
        requirements = fp.read()
    requirements = adjust(requirements, torch_version)
    logging.info(requirements)  # on purpose - to debug
    with open(requirements_path, "w") as fp:
        fp.write(requirements)
