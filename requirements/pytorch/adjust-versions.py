import os
import re
from typing import Dict, Optional

# IMPORTANT: this list needs to be sorted in reverse
VERSIONS = [
    dict(torch="2.0.0", torchvision="0.15.0"),  # nightly
    dict(torch="1.13.1", torchvision="0.14.1"),  # stable
    dict(torch="1.13.0", torchvision="0.14.0"),
    dict(torch="1.12.1", torchvision="0.13.1"),
    dict(torch="1.12.0", torchvision="0.13.0"),
    dict(torch="1.11.0", torchvision="0.12.0"),
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


def replace(req: str, torch_version: Optional[str] = None, remove_torch_ver: bool = False) -> str:
    if not torch_version:
        import torch

        torch_version = torch.__version__
    assert remove_torch_ver or torch_version, f"invalid torch: {torch_version}"

    # remove comments and strip whitespace
    req = re.sub(rf"\s*#.*{os.linesep}", os.linesep, req).strip()

    latest = find_latest(torch_version)
    for lib, version in latest.items():
        if remove_torch_ver and lib == "torch":
            replace = lib
        else:
            replace = f"{lib}=={version}" if version else ""
        req = re.sub(rf"\b{lib}(?!\w).*", replace, req)

    return req


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("requirements_path", type=str)
    parser.add_argument("torch_version", type=str, default=None)
    parser.add_argument("--remove_torch_ver", action="store_true")
    args = parser.parse_args()
    print(f"requirements_path='{args.requirements_path}' with torch_version='{args.torch_version}'")

    with open(args.requirements_path) as fp:
        requirements = fp.read()
    requirements = replace(requirements, args.torch_version, args.remove_torch_ver)
    print(requirements)  # on purpose - to debug
    with open(args.requirements_path, "w") as fp:
        fp.write(requirements)
