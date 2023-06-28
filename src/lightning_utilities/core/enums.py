# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import warnings
from enum import Enum
from typing import List, Optional

from typing_extensions import Literal


class StrEnum(str, Enum):
    """Type of any enumerator with allowed comparison to string invariant to cases.

    >>> class MySE(StrEnum):
    ...     t1 = "T-1"
    ...     t2 = "T-2"
    >>> MySE("T-1") == MySE.t1
    True
    >>> MySE.from_str("t-2", source="value") == MySE.t2
    True
    >>> MySE.from_str("t-2", source="value")
    <MySE.t2: 'T-2'>
    >>> MySE.from_str("t-3", source="any")
    Traceback (most recent call last):
      ...
    ValueError: Invalid match: expected one of ['t1', 't2', 'T-1', 'T-2'], but got t-3.
    """

    @classmethod
    def from_str(cls, value: str, source: Literal["key", "value", "any"] = "key") -> "StrEnum":
        """Create ``StrEnum`` from a string matching the key or value.

        Args:
            value: matching string
            source: compare with:

                - ``"key"``: validates only from the enum keys, typical alphanumeric with "_"
                - ``"value"``: validates only from the values, could be any string
                - ``"any"``: validates with any key or value, but key has priority

        Raises:
            ValueError:
                if requested string does not match any option based on selected source.
        """
        if source in ("key", "any"):
            for enum_key in cls.__members__:
                if enum_key.lower() == value.lower():
                    return cls[enum_key]
        if source in ("value", "any"):
            for enum_key, enum_val in cls.__members__.items():
                if enum_val == value:
                    return cls[enum_key]
        raise ValueError(f"Invalid match: expected one of {cls._allowed_matches(source)}, but got {value}.")

    @classmethod
    def try_from_str(cls, value: str, source: Literal["key", "value", "any"] = "key") -> Optional["StrEnum"]:
        """Try to create emun and if it does not match any, return `None`."""
        try:
            return cls.from_str(value, source)
        except ValueError:
            warnings.warn(  # noqa: B028
                UserWarning(f"Invalid string: expected one of {cls._allowed_matches(source)}, but got {value}.")
            )
        return None

    @classmethod
    def _allowed_matches(cls, source: str) -> List[str]:
        keys, vals = [], []
        for enum_key, enum_val in cls.__members__.items():
            keys.append(enum_key)
            vals.append(enum_val.value)
        if source == "key":
            return keys
        if source == "value":
            return vals
        return keys + vals

    def __eq__(self, other: object) -> bool:
        """Compare two instances."""
        if isinstance(other, Enum):
            other = other.value
        return self.value.lower() == str(other).lower()

    def __hash__(self) -> int:
        """Return unique hash."""
        # re-enable hashtable, so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.value.lower())
