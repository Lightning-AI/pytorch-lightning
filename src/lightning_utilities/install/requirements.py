# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
"""Utilities to parse and adjust Python requirements files.

This module parses requirement lines while preserving inline comments and pip arguments and
supports relaxing version pins based on a chosen unfreeze strategy: "none", "major", or "all".

"""

import re
from collections.abc import Iterable, Iterator
from distutils.version import LooseVersion
from pathlib import Path
from typing import Any, Optional, Union

from pkg_resources import Requirement, yield_lines  # type: ignore[import-untyped]


class _RequirementWithComment(Requirement):
    """Requirement subclass that preserves an inline comment and optional pip argument.

    Attributes:
        comment: The trailing comment captured from the requirement line (including the leading '# ...').
        pip_argument: A preceding pip argument line (e.g., ``"--extra-index-url ..."``) associated
            with this requirement, or ``None`` if not provided.
        strict: Whether the special marker ``"# strict"`` appears in ``comment`` (case-insensitive), in which case
            upper bound adjustments are disabled.

    """

    strict_string = "# strict"

    def __init__(self, *args: Any, comment: str = "", pip_argument: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.comment = comment
        if not (pip_argument is None or pip_argument):  # sanity check that it's not an empty str
            raise RuntimeError(f"wrong pip argument: {pip_argument}")
        self.pip_argument = pip_argument
        self.strict = self.strict_string in comment.lower()

    def adjust(self, unfreeze: str) -> str:
        """Adjust version specifiers according to the selected unfreeze strategy.

        The special marker ``"# strict"`` in the captured comment disables any relaxation of upper bounds.

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

        Args:
            unfreeze: One of:
                - ``"none"``: Keep all version specifiers unchanged.
                - ``"major"``: Relax the upper bound to the next major version (e.g., ``<2.0``).
                - ``"all"``: Drop any upper bound constraint entirely.

        Returns:
            The adjusted requirement string. If strict, the original string is returned with the strict marker appended.

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
    r"""Adapted from ``pkg_resources.parse_requirements`` to include comments and pip arguments.

    Parses a sequence or string of requirement lines, preserving trailing comments and associating any
    preceding pip arguments (``--...``) with the subsequent requirement. Lines starting with ``-r`` or
    containing direct URLs are ignored.

    >>> txt = ['# ignored', '', 'this # is an', '--piparg', 'example', 'foo # strict', 'thing', '-r different/file.txt']
    >>> [r.adjust('none') for r in _parse_requirements(txt)]
    ['this', 'example', 'foo  # strict', 'thing']
    >>> txt = '\\n'.join(txt)
    >>> [r.adjust('none') for r in _parse_requirements(txt)]
    ['this', 'example', 'foo  # strict', 'thing']

    Args:
        strs: Either an iterable of requirement lines or a single multi-line string.

    Yields:
        _RequirementWithComment: Parsed requirement objects with preserved comment and pip argument.

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


def load_requirements(path_dir: str, file_name: str = "base.txt", unfreeze: str = "all") -> list[str]:
    """Load, parse, and optionally relax requirement specifiers from a file.

    >>> import os
    >>> from lightning_utilities import _PROJECT_ROOT
    >>> path_req = os.path.join(_PROJECT_ROOT, "requirements")
    >>> load_requirements(path_req, "docs.txt", unfreeze="major")  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['sphinx<6.0,>=4.0', ...]

    Args:
        path_dir: Directory containing the requirements file.
        file_name: The requirements filename inside ``path_dir``.
        unfreeze: Unfreeze strategy: ``"none"``, ``"major"``, or ``"all"`` (see ``_RequirementWithComment.adjust``).

    Returns:
        A list of requirement strings adjusted according to ``unfreeze``.

    Raises:
        ValueError: If ``unfreeze`` is not one of the supported options.
        FileNotFoundError: If the composed path does not exist.

    """
    if unfreeze not in {"none", "major", "all"}:
        raise ValueError(f'unsupported option of "{unfreeze}"')
    path = Path(path_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(f"missing file for {(path_dir, file_name, path)}")
    text = path.read_text()
    return [req.adjust(unfreeze) for req in _parse_requirements(text)]
