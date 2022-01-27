from abc import ABC, abstractmethod


class Executor(ABC):
    def __init__(self, strategy):
        self.strategy = strategy

    @abstractmethod
    def execute(self, trainer, fn, *args, **kwargs) -> bool:
        """Executes the proceses."""
