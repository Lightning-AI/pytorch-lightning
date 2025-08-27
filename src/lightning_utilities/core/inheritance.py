# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from collections.abc import Iterator


def get_all_subclasses_iterator(cls: type) -> Iterator[type]:
    """Depth-first iterator over all subclasses of ``cls`` (recursively)."""

    def recurse(cl: type) -> Iterator[type]:
        for subclass in cl.__subclasses__():
            yield subclass
            yield from recurse(subclass)

    yield from recurse(cls)


def get_all_subclasses(cls: type) -> set[type]:
    """Return a set containing all subclasses of ``cls`` discovered recursively."""
    return set(get_all_subclasses_iterator(cls))
