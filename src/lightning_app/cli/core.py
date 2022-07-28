import abc

from rich.table import Table


class Formatable(abc.ABC):
    @abc.abstractmethod
    def as_table(self) -> Table:
        pass

    @abc.abstractmethod
    def as_json(self) -> str:
        pass
