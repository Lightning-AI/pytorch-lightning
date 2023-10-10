# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Adjusting version across PTorch ecosystem."""
import logging
import os
import re
import sys
from typing import Dict, List, Optional

VERSIONS = [
    {"torch": "2.2.0", "torchvision": "0.17.0", "torchtext": "0.17.0", "torchaudio": "2.2.0"},  # nightly
    {"torch": "2.1.0", "torchvision": "0.16.0", "torchtext": "0.16.0", "torchaudio": "2.1.0"},  # stable
    {"torch": "2.0.1", "torchvision": "0.15.2", "torchtext": "0.15.2", "torchaudio": "2.0.2"},
    {"torch": "2.0.0", "torchvision": "0.15.1", "torchtext": "0.15.1", "torchaudio": "2.0.1"},
    {"torch": "1.14.0", "torchvision": "0.15.0", "torchtext": "0.15.0", "torchaudio": "0.14.0"},  # nightly / shifted
    {"torch": "1.13.1", "torchvision": "0.14.1", "torchtext": "0.14.1", "torchaudio": "0.13.1"},
    {"torch": "1.13.0", "torchvision": "0.14.0", "torchtext": "0.14.0", "torchaudio": "0.13.0"},
    {"torch": "1.12.1", "torchvision": "0.13.1", "torchtext": "0.13.1", "torchaudio": "0.12.1"},
    {"torch": "1.12.0", "torchvision": "0.13.0", "torchtext": "0.13.0", "torchaudio": "0.12.0"},
    {"torch": "1.11.0", "torchvision": "0.12.0", "torchtext": "0.12.0", "torchaudio": "0.11.0"},
    {"torch": "1.10.2", "torchvision": "0.11.3", "torchtext": "0.11.2", "torchaudio": "0.10.2"},
    {"torch": "1.10.1", "torchvision": "0.11.2", "torchtext": "0.11.1", "torchaudio": "0.10.1"},
    {"torch": "1.10.0", "torchvision": "0.11.1", "torchtext": "0.11.0", "torchaudio": "0.10.0"},
    {"torch": "1.9.1", "torchvision": "0.10.1", "torchtext": "0.10.1", "torchaudio": "0.9.1"},
    {"torch": "1.9.0", "torchvision": "0.10.0", "torchtext": "0.10.0", "torchaudio": "0.9.0"},
    {"torch": "1.8.2", "torchvision": "0.9.1", "torchtext": "0.9.1", "torchaudio": "0.8.1"},
    {"torch": "1.8.1", "torchvision": "0.9.1", "torchtext": "0.9.1", "torchaudio": "0.8.1"},
    {"torch": "1.8.0", "torchvision": "0.9.0", "torchtext": "0.9.0", "torchaudio": "0.8.0"},
]


def find_latest(ver: str) -> Dict[str, str]:
    """Find the latest version."""
    # drop all except semantic version
    ver = re.search(r"([\.\d]+)", ver).groups()[0]
    # in case there remaining dot at the end - e.g "1.9.0.dev20210504"
    ver = ver[:-1] if ver[-1] == "." else ver
    logging.debug(f"finding ecosystem versions for: {ver}")

    # find first match
    for option in VERSIONS:
        if option["torch"].startswith(ver):
            return option

    raise ValueError(f"Missing {ver} in {VERSIONS}")


def adjust(requires: List[str], pytorch_version: Optional[str] = None) -> List[str]:
    """Adjust the versions to be paired within pytorch ecosystem."""
    if not pytorch_version:
        import torch

        pytorch_version = torch.__version__
    if not pytorch_version:
        raise ValueError(f"invalid torch: {pytorch_version}")

    requires_ = []
    options = find_latest(pytorch_version)
    logging.debug(f"determined ecosystem alignment: {options}")
    for req in requires:
        req_split = req.strip().split("#", maxsplit=1)
        # anything before fst # shall be requirements
        req = req_split[0].strip()
        # anything after # in the line is comment
        comment = "" if len(req_split) < 2 else "  #" + req_split[1]
        if not req:
            # if only comment make it short
            requires_.append(comment.strip())
            continue
        for lib, version in options.items():
            replace = f"{lib}=={version}" if version else ""
            req = re.sub(rf"\b{lib}(?![-_\w]).*", replace, req)
        requires_.append(req + comment.rstrip())

    return requires_


def _offset_print(reqs: List[str], offset: str = "\t|\t") -> str:
    """Adding offset to each line for the printing requirements."""
    reqs = [offset + r for r in reqs]
    return os.linesep.join(reqs)


def main(requirements_path: str, torch_version: Optional[str] = None) -> None:
    """The main entry point with mapping to the CLI for positional arguments only."""
    # rU - universal line ending - https://stackoverflow.com/a/2717154/4521646
    with open(requirements_path, encoding="utf8") as fopen:
        requirements = fopen.readlines()
    requirements = adjust(requirements, torch_version)
    logging.info(
        f"requirements_path='{requirements_path}' with arg torch_version='{torch_version}' >>\n"
        f"{_offset_print(requirements)}"
    )
    with open(requirements_path, "w", encoding="utf8") as fopen:
        fopen.writelines([r + os.linesep for r in requirements])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        from fire import Fire

        Fire(main)
    except (ModuleNotFoundError, ImportError):
        main(*sys.argv[1:])
