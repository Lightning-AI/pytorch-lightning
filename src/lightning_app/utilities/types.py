import typing as t

from typing_extensions import Protocol, runtime_checkable

from lightning_app import LightningFlow, LightningWork
from lightning_app.structures import Dict, List

Component = t.Union[LightningFlow, LightningWork, Dict, List]
ComponentTuple = (LightningFlow, LightningWork, Dict, List)


@runtime_checkable
class Hashable(Protocol):
    def to_dict(self) -> t.Dict[str, t.Any]:
        ...
