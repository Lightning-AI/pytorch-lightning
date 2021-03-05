# stdlib
import os
import platform
import re
import sys
from typing import Any, Dict

VERSIONS_NONE: Dict[str, Any] = dict(torchvision=None, torchcsprng=None)
VERSIONS_LUT: Dict[str, Dict[str, Any]] = {
    "1.4.0": dict(torchvision="0.5.0", torchtext="0.5"),
    "1.5.0": dict(torchvision="0.6.0", torchtext="0.6"),
    "1.5.1": dict(torchvision="0.6.1", torchtext="0.6"),
    "1.6.0": dict(torchvision="0.7.0", torchtext="0.7"),
    "1.7.0": dict(torchvision="0.8.1", torchtext="0.8"),
    "1.7.1": dict(torchvision="0.8.2", torchtext="0.8.1"),
    "1.8.0": dict(torchvision="0.9.0", torchtext="0.9"),
}

system = platform.system()


def main(path_req: str, torch_version: str = None) -> None:
    with open(path_req, "r") as fp:
        req = fp.read()

    if not torch_version:
        import torch
        torch_version = torch.__version__

    dep_versions = VERSIONS_LUT.get(torch_version, VERSIONS_NONE)
    dep_versions["torch"] = torch_version
    for lib in dep_versions:
        version = dep_versions[lib]
        replace = f"{lib}=={version}\n"
        req = re.sub(rf"{lib}[>=]*[\d\.]*{os.linesep}", replace, req)

    with open(path_req, "w") as fp:
        fp.write(req)


if __name__ == "__main__":
    main(*sys.argv[1:])
