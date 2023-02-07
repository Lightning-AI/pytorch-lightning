# Copyright The PyTorch Lightning team.
# Licensed under the Apache License, Version 2.0 (the "License");
#     http://www.apache.org/licenses/LICENSE-2.0
#
import warnings
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
    >>> MySE.from_str("t-2", source="value") == MySE.t2
    True
    """

    @classmethod
    def from_str(
        cls, value: str, source: Literal["key", "value", "any"] = "key", strict: bool = False
    ) -> Optional["StrEnum"]:
        """Create StrEnum from a sting matching the key or value.

        Args:
            value: matching string
            source: compare with:

                - ``"key"``: validates only with Enum keys, typical alphanumeric with "_"
                - ``"value"``: validates only with Enum values, could be any string
                - ``"key"``: validates with any key or value, but key has priority

            strict: allow not matching string and returns None; if false raises exceptions

        Raises:
            ValueError:
                if requested string does not match any option based on selected source and use ``"strict=True"``
            UserWarning:
                if requested string does not match any option based on selected source and use ``"strict=False"``

        Example:
            >>> class MySE(StrEnum):
            ...     t1 = "T-1"
            ...     t2 = "T-2"
            >>> MySE.from_str("t-1", source="key")
            >>> MySE.from_str("t-2", source="value")
            <MySE.t2: 'T-2'>
            >>> MySE.from_str("t-3", source="any", strict=True)
            Traceback (most recent call last):
              ...
            ValueError: Invalid match: expected one of ['t1', 't2', 'T-1', 'T-2'], but got t-3.
        """
        allowed = cls._allowed_matches(source)
        if strict and not any(enum_.lower() == value.lower() for enum_ in allowed):
            raise ValueError(f"Invalid match: expected one of {allowed}, but got {value}.")

        if source in ("key", "any"):
            for enum_key in cls.__members__.keys():
                if enum_key.lower() == value.lower():
                    return cls[enum_key]
        if source in ("value", "any"):
            for enum_key, enum_val in cls.__members__.items():
                if enum_val == value:
                    return cls[enum_key]

        warnings.warn(UserWarning(f"Invalid string: expected one of {allowed}, but got {value}."))
        return None

    @classmethod
    def _allowed_matches(cls, source: str) -> list:
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
        if isinstance(other, Enum):
            other = other.value
        return self.value.lower() == str(other).lower()

    def __hash__(self) -> int:
        # re-enable hashtable, so it can be used as a dict key or in a set
        # example: set(LightningEnum)
        return hash(self.value.lower())
