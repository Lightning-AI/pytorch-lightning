from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDatabaseClient(ABC):
    @abstractmethod
    def select_all(self):
        ...

    @abstractmethod
    def insert(self):
        ...

    @abstractmethod
    def update(self):
        ...

    @abstractmethod
    def delete(self):
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        ...
