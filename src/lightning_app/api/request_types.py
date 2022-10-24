from dataclasses import asdict, dataclass
from typing import Any, Optional

from deepdiff import Delta


@dataclass
class BaseRequest:
    def to_dict(self):
        return asdict(self)


@dataclass
class DeltaRequest(BaseRequest):
    delta: Delta

    def to_dict(self):
        return self.delta.to_dict()


@dataclass
class CommandRequest(BaseRequest):
    id: str
    name: str
    method_name: str
    args: Any
    kwargs: Any


@dataclass
class APIRequest(BaseRequest):
    id: str
    name: str
    method_name: str
    args: Any
    kwargs: Any


@dataclass
class RequestResponse(BaseRequest):
    status_code: int
    content: Optional[str] = None
