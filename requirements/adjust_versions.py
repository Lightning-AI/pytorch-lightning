import os
import re
import sys
from typing import Dict, Optional

VERSIONS = [
    dict(torch="1.9.0", torchvision="", torchtext=""),  # nightly
    dict(torch="1.8.1", torchvision="0.9.1", torchtext="0.9.1"),
    dict(torch="1.8.0", torchvision="0.9.0", torchtext="0.9.0"),
    dict(torch="1.7.1", torchvision="0.8.2", torchtext="0.8.1"),
    dict(torch="1.7.0", torchvision="0.8.1", torchtext="0.8.0"),
    dict(torch="1.6.0", torchvision="0.7.0", torchtext="0.7"),
    dict(torch="1.5.1", torchvision="0.6.1", torchtext="0.6"),
    dict(torch="1.5.0", torchvision="0.6.0", torchtext="0.6"),
    dict(torch="1.4.0", torchvision="0.5.0", torchtext="0.5"),
]
VERSIONS.sort(key=lambda v: v["torch"], reverse=True)


def find_latest(ver: str) -> Dict[str, str]:
    # drop all except semantic version
    ver = re.search(r'([\.\d]+)', ver).groups()[0]
    # in case there remaining dot at the end - e.g "1.9.0.dev20210504"
    ver = ver[:-1] if ver[-1] == '.' else ver
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

    with open(path_req, "r") as fp:
        req = fp.read()

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
