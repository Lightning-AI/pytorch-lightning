# Copyright The Lightning AI team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from collections.abc import Iterator


def get_all_subclasses_iterator(cls: type) -> Iterator[type]:
    """Iterate over all subclasses."""

    def recurse(cl: type) -> Iterator[type]:
        for subclass in cl.__subclasses__():
            yield subclass
            yield from recurse(subclass)

    yield from recurse(cls)


def get_all_subclasses(cls: type) -> set[type]:
    """List all subclasses of a class."""
    return set(get_all_subclasses_iterator(cls))
