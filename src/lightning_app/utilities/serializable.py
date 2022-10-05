from typing import Any, Dict

from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Hashable(Protocol):
    def to_dict(self) -> Dict[str, Any]:
        ...
