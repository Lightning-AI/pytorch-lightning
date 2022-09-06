# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from enum import Enum
from typing import Optional


class StrEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases."""

    @classmethod
    def from_str(cls, value: str) -> Optional["StrEnum"]:
        statuses = cls.__members__.keys()
        for st in statuses:
            if st.lower() == value.lower():
                return cls[st]
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Enum):
            other = other.value
        return self.value.lower() == str(other).lower()

    def __hash__(self) -> int:
        # re-enable hashtable so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.value.lower())
