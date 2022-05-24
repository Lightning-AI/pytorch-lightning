import os
import re
import sys
from typing import Dict, Optional

# IMPORTANT: this list needs to be sorted in reverse
VERSIONS = [
    dict(torch="1.12.0", torchvision="0.12.*", torchtext=""),  # nightly
    dict(torch="1.11.0", torchvision="0.12.0", torchtext="0.12.0"),  # pre-release
    dict(torch="1.10.2", torchvision="0.11.3", torchtext="0.11.2"),  # stable
    dict(torch="1.10.1", torchvision="0.11.2", torchtext="0.11.1"),
    dict(torch="1.10.0", torchvision="0.11.1", torchtext="0.11.0"),
    dict(torch="1.9.1", torchvision="0.10.1", torchtext="0.10.1"),
    dict(torch="1.9.0", torchvision="0.10.0", torchtext="0.10.0"),
    # dict(torch="1.8.2", torchvision="0.9.1", torchtext="0.9.1"), # LTS # Not on PyPI, commented so 1.8.1 is used
    dict(torch="1.8.1", torchvision="0.9.1", torchtext="0.9.1"),
    dict(torch="1.8.0", torchvision="0.9.0", torchtext="0.9.0"),
]


def find_latest(ver: str) -> Dict[str, str]:
    # drop all except semantic version
    ver = re.search(r"([\.\d]+)", ver).groups()[0]
    # in case there remaining dot at the end - e.g "1.9.0.dev20210504"
    ver = ver[:-1] if ver[-1] == "." else ver
    print(f"finding ecosystem versions for: {ver}")

    # find first match
    for option in VERSIONS:
        if option["torch"].startswith(ver):
            return option

    raise ValueError(f"Missing {ver} in {VERSIONS}")


def main(req: str, torch_version: Optional[str] = None) -> str:
    if not torch_version:
        import torch

        torch_version = torch.__version__
    assert torch_version, f"invalid torch: {torch_version}"

    # remove comments and strip whitespace
    req = re.sub(rf"\s*#.*{os.linesep}", os.linesep, req).strip()

    latest = find_latest(torch_version)
    for lib, version in latest.items():
        replace = f"{lib}=={version}" if version else ""
        req = re.sub(rf"\b{lib}(?!\w).*", replace, req)

    return req


def test():
    requirements = """
    torch>=1.2.*
    torch==1.2.3
    torch==1.4
    torch
    future>=0.17.1
    pytorch==1.5.6+123dev0
    torchvision
    torchmetrics>=0.4.1
    """
    expected = """
    torch==1.9.1
    torch==1.9.1
    torch==1.9.1
    torch==1.9.1
    future>=0.17.1
    pytorch==1.5.6+123dev0
    torchvision==0.10.1
    torchmetrics>=0.4.1
    """.strip()
    actual = main(requirements, "1.9")
    assert actual == expected, (actual, expected)


if __name__ == "__main__":
    test()  # sanity check

    if len(sys.argv) == 3:
        requirements_path, torch_version = sys.argv[1:]
    else:
        requirements_path, torch_version = sys.argv[1], None

    with open(requirements_path) as fp:
        requirements = fp.read()
    requirements = main(requirements, torch_version)
    print(requirements)  # on purpose - to debug
    with open(requirements_path, "w") as fp:
        fp.write(requirements)
