import os
import re
import sys
from typing import Dict, Optional

# IMPORTANT: this list needs to be sorted in reverse
VERSIONS = [
    dict(torch="1.11.0", torchvision="0.11.*", torchtext=""),  # nightly
    dict(torch="1.10.0", torchvision="0.11.*", torchtext=""),
    dict(torch="1.9.1", torchvision="0.10.1", torchtext="0.10.1"),
    dict(torch="1.9.0", torchvision="0.10.0", torchtext="0.10.0"),
    dict(torch="1.8.2", torchvision="0.9.1", torchtext="0.9.1"),
    dict(torch="1.8.1", torchvision="0.9.1", torchtext="0.9.1"),
    dict(torch="1.8.0", torchvision="0.9.0", torchtext="0.9.0"),
    dict(torch="1.7.1", torchvision="0.8.2", torchtext="0.8.1"),
    dict(torch="1.7.0", torchvision="0.8.1", torchtext="0.8.0"),
    dict(torch="1.6.0", torchvision="0.7.0", torchtext="0.7"),
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


def main(path_req: str, torch_version: Optional[str] = None) -> None:
    if not torch_version:
        import torch

        torch_version = torch.__version__
    assert torch_version, f"invalid torch: {torch_version}"

    with open(path_req) as fp:
        req = fp.read()
    # remove comments
    req = re.sub(rf"\s*#.*{os.linesep}", os.linesep, req)

    latest = find_latest(torch_version)
    for lib, version in latest.items():
        replace = f"{lib}=={version}" if version else lib
        replace += os.linesep
        req = re.sub(rf"{lib}[>=]*[\d\.]*{os.linesep}", replace, req)

    print(req)  # on purpose - to debug
    with open(path_req, "w") as fp:
        fp.write(req)


if __name__ == "__main__":
    main(*sys.argv[1:])
