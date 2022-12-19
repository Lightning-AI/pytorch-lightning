# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
from enum import Enum
from typing import Optional

from typing_extensions import Literal


class StrEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases.

    >>> class MySE(StrEnum):
    ...     t1 = "T-1"
    ...     t2 = "T-2"
    >>> MySE("T-1") == MySE.t1
    True
    >>> MySE.from_str("t-2") == MySE.t2
    True
    """

    @classmethod
    def from_str(cls, value: str, source: Literal["key", "value", "any"] = "key") -> Optional["StrEnum"]:
        for st, val in cls.__members__.items():
            if source in ("key", "any") and st.lower() == value.lower():
                return cls[st]
            if source in ("value", "any") and val.lower() == value.lower():
                return cls[st]
        return None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Enum):
            other = other.value
        return self.value.lower() == str(other).lower()

    def __hash__(self) -> int:
        # re-enable hashtable, so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.value.lower())
