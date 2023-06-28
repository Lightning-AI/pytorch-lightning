# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import re
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Optional, Union

from pkg_resources import Requirement, yield_lines


class _RequirementWithComment(Requirement):
    strict_string = "# strict"

    def __init__(self, *args: Any, comment: str = "", pip_argument: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.comment = comment
        if not (pip_argument is None or pip_argument):  # sanity check that it's not an empty str
            raise RuntimeError(f"wrong pip argument: {pip_argument}")
        self.pip_argument = pip_argument
        self.strict = self.strict_string in comment.lower()

    def adjust(self, unfreeze: str) -> str:
        """Remove version restrictions unless they are strict.

        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# anything").adjust("none")
        'arrow<=1.2.2,>=1.2.0'
        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# strict").adjust("none")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow<=1.2.2,>=1.2.0", comment="# my name").adjust("all")
        'arrow>=1.2.0'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust("all")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow").adjust("all")
        'arrow'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# cool").adjust("major")
        'arrow<2.0,>=1.2.0'
        >>> _RequirementWithComment("arrow>=1.2.0, <=1.2.2", comment="# strict").adjust("major")
        'arrow<=1.2.2,>=1.2.0  # strict'
        >>> _RequirementWithComment("arrow>=1.2.0").adjust("major")
        'arrow>=1.2.0'
        >>> _RequirementWithComment("arrow").adjust("major")
        'arrow'
        """
        out = str(self)
        if self.strict:
            return f"{out}  {self.strict_string}"
        if unfreeze == "major":
            for operator, version in self.specs:
                if operator in ("<", "<="):
                    major = LooseVersion(version).version[0]
                    # replace upper bound with major version increased by one
                    return out.replace(f"{operator}{version}", f"<{int(major) + 1}.0")
        elif unfreeze == "all":
            for operator, version in self.specs:
                if operator in ("<", "<="):
                    # drop upper bound
                    return out.replace(f"{operator}{version},", "")
        elif unfreeze != "none":
            raise ValueError(f"Unexpected unfreeze: {unfreeze!r} value.")
        return out


def _parse_requirements(strs: Union[str, Iterable[str]]) -> Iterator[_RequirementWithComment]:
    r"""Adapted from `pkg_resources.parse_requirements` to include comments.

    >>> txt = ['# ignored', '', 'this # is an', '--piparg', 'example', 'foo # strict', 'thing', '-r different/file.txt']
    >>> [r.adjust('none') for r in _parse_requirements(txt)]
    ['this', 'example', 'foo  # strict', 'thing']
    >>> txt = '\\n'.join(txt)
    >>> [r.adjust('none') for r in _parse_requirements(txt)]
    ['this', 'example', 'foo  # strict', 'thing']
    """
    lines = yield_lines(strs)
    pip_argument = None
    for line in lines:
        # Drop comments -- a hash without a space may be in a URL.
        if " #" in line:
            comment_pos = line.find(" #")
            line, comment = line[:comment_pos], line[comment_pos:]
        else:
            comment = ""
        # If there is a line continuation, drop it, and append the next line.
        if line.endswith("\\"):
            line = line[:-2].strip()
            try:
                line += next(lines)
            except StopIteration:
                return
        # If there's a pip argument, save it
        if line.startswith("--"):
            pip_argument = line
            continue
        if line.startswith("-r "):
            # linked requirement files are unsupported
            continue
        if "@" in line or re.search("https?://", line):
            # skip lines with links like `pesq @ git+https://github.com/ludlows/python-pesq`
            continue
        yield _RequirementWithComment(line, comment=comment, pip_argument=pip_argument)
        pip_argument = None


def load_requirements(path_dir: str, file_name: str = "base.txt", unfreeze: str = "all") -> List[str]:
    """Load requirements from a file.

    >>> import os
    >>> from lightning_utilities import _PROJECT_ROOT
    >>> path_req = os.path.join(_PROJECT_ROOT, "requirements")
    >>> load_requirements(path_req, "docs.txt", unfreeze="major")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sphinx<6.0,>=4.0', ...]
    """
    if unfreeze not in {"none", "major", "all"}:
        raise ValueError(f'unsupported option of "{unfreeze}"')
    path = Path(path_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(f"missing file for {(path_dir, file_name, path)}")
    text = path.read_text()
    return [req.adjust(unfreeze) for req in _parse_requirements(text)]
