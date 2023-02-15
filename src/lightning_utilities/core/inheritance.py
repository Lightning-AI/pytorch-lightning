# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from typing import Iterator, Set, Type


def get_all_subclasses_iterator(cls: Type) -> Iterator[Type]:
    """Iterate over all subclasses."""

    def recurse(cl: Type) -> Iterator[Type]:
        for subclass in cl.__subclasses__():
            yield subclass
            yield from recurse(subclass)

    yield from recurse(cls)


def get_all_subclasses(cls: Type) -> Set[Type]:
    """List all subclasses of a class."""
    return set(get_all_subclasses_iterator(cls))
