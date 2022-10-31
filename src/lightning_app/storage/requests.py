from dataclasses import dataclass
from typing import Optional


@dataclass
class _GetRequest:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""


@dataclass
class _GetResponse:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""
    exception: Optional[Exception] = None
    timedelta: Optional[float] = None


@dataclass
class _ExistsRequest:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""


@dataclass
class _ExistsResponse:
    source: str
    name: str
    path: str
    hash: str
    destination: str = ""
    exists: Optional[bool] = None
    timedelta: Optional[float] = None
