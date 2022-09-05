from typing import Iterable

from lightning_utilities.core.imports import package_available

if package_available("torch"):
    import torch
else:
    # minimal torch implementation to avoid installing torch in testing CI
    class TensorMock:
        def __init__(self, data):
            self.data = data

        def __add__(self, other):
            if isinstance(self.data, Iterable):
                if isinstance(other, (int, float)):
                    return TensorMock([a + other for a in self.data])
                if isinstance(other, Iterable):
                    return TensorMock([a + b for a, b in zip(self, other)])
            return self.data + other

        def __mul__(self, other):
            if isinstance(self.data, Iterable):
                if isinstance(other, (int, float)):
                    return TensorMock([a * other for a in self.data])
                if isinstance(other, Iterable):
                    return TensorMock([a * b for a, b in zip(self, other)])
            return self.data * other

        def __iter__(self):
            return iter(self.data)

        def __repr__(self):
            return repr(self.data)

        def __eq__(self, other):
            return self.data == other

    class TorchMock:
        Tensor = TensorMock

        @staticmethod
        def tensor(data):
            return TensorMock(data)

        @staticmethod
        def equal(a, b):
            return a == b

        @staticmethod
        def arange(*args):
            return TensorMock(list(range(*args)))

    torch = TorchMock()
