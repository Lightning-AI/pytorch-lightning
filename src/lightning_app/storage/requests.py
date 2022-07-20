from dataclasses import dataclass
from typing import Optional


@dataclass
class GetRequest:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""


@dataclass
class GetResponse:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""
    exception: Optional[Exception] = None
    timedelta: Optional[float] = None


@dataclass
class ExistsRequest:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""


@dataclass
class ExistsResponse:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""
    exists: Optional[bool] = None
    timedelta: Optional[float] = None
