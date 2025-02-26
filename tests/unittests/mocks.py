from collections.abc import Iterable
from typing import Any

from lightning_utilities.core.imports import package_available

if package_available("torch"):
    import torch
else:
    # minimal torch implementation to avoid installing torch in testing CI
    class TensorMock:
        def __init__(self, data) -> None:
            self.data = data

        def __add__(self, other):
            """Perform and operation."""
            if isinstance(self.data, Iterable):
                if isinstance(other, (int, float)):
                    return TensorMock([a + other for a in self.data])
                if isinstance(other, Iterable):
                    return TensorMock([a + b for a, b in zip(self, other)])
            return self.data + other

        def __mul__(self, other):
            """Perform mul operation."""
            if isinstance(self.data, Iterable):
                if isinstance(other, (int, float)):
                    return TensorMock([a * other for a in self.data])
                if isinstance(other, Iterable):
                    return TensorMock([a * b for a, b in zip(self, other)])
            return self.data * other

        def __iter__(self):
            """Iterate."""
            return iter(self.data)

        def __repr__(self) -> str:
            """Return object representation."""
            return repr(self.data)

        def __eq__(self, other):
            """Perform equal operation."""
            return self.data == other

        def add_(self, value):
            self.data += value
            return self.data

    class TorchMock:
        Tensor = TensorMock

        @staticmethod
        def tensor(data: Any) -> TensorMock:
            return TensorMock(data)

        @staticmethod
        def equal(a: Any, b: Any) -> bool:
            return a == b

        @staticmethod
        def arange(*args: Any) -> TensorMock:
            return TensorMock(list(range(*args)))

    torch = TorchMock()
