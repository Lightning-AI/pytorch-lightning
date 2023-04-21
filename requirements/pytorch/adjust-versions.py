import os
import re
import sys
from typing import Dict, Optional

# IMPORTANT: this list needs to be sorted in reverse
VERSIONS = [
    {"torch": "2.1.0", "torchvision": "0.16.0"},  # nightly
    {"torch": "2.0.0", "torchvision": "0.15.1"},  # stable
    {"torch": "1.13.1", "torchvision": "0.14.1"},
    {"torch": "1.13.0", "torchvision": "0.14.0"},
    {"torch": "1.12.1", "torchvision": "0.13.1"},
    {"torch": "1.12.0", "torchvision": "0.13.0"},
    {"torch": "1.11.0", "torchvision": "0.12.0"},
]


def find_latest(ver: str) -> Dict[str, str]:
    # drop all except semantic version
    ver = re.search(r"([\.\d]+)", ver).groups()[0]
    # in case there remaining dot at the end - e.g "1.9.0.dev20210504"
    ver = ver[:-1] if ver[-1] == "." else ver
    print(f"\n\n\nfinding ecosystem versions for: {ver}")

    # find first match
    for option in VERSIONS:
        if option["torch"].startswith(ver):
            return option

    raise ValueError(f"Missing {ver} in {VERSIONS}")


def replace(req: str, torch_version: Optional[str] = None) -> str:
    if not torch_version:
        import torch

        torch_version = torch.__version__
    assert torch_version, f"invalid torch: {torch_version}"  # noqa: S101

    # remove comments and strip whitespace
    req = re.sub(rf"\s*#.*{os.linesep}", os.linesep, req).strip()

    latest = find_latest(torch_version)
    for lib, version in latest.items():
        replace = f"{lib}=={version}" if version else ""
        req = re.sub(rf"\b{lib}(?!\w).*", replace, req)

    return req


if __name__ == "__main__":
    if len(sys.argv) == 3:
        requirements_path, torch_version = sys.argv[1:]
    else:
        requirements_path, torch_version = sys.argv[1], None
    print(f"requirements_path='{requirements_path}' with torch_version='{torch_version}'")

    with open(requirements_path) as fp:
        requirements = fp.read()
    requirements = replace(requirements, torch_version)
    print(requirements)  # on purpose - to debug
    with open(requirements_path, "w") as fp:
        fp.write(requirements)
