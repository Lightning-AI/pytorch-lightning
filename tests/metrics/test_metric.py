import pytest
import torch
from pytorch_lightning.metrics.metric import Metric
import os

torch.manual_seed(42)


class Dummy(Metric):
    name = "Dummy"

    def __init__(self):
        super().__init__()
        self.add_state("x", 0, reduction=False)

    def update(self):
        pass

    def compute(self):
        pass


def test_inherit():
    a = Dummy()


def test_reset():
    class A(Dummy):
        pass

    a = A()
    assert a.x == 0
    a.x = 5
    a.reset()
    assert a.x == 0


def test_update():
    class A(Dummy):
        def update(self, x):
            self.x += x

    a = A()
    assert a.x == 0
    a.update(1)
    assert a.x == 1
    a.update(2)
    assert a.x == 3


def test_compute():
    class A(Dummy):
        def update(self, x):
            self.x += x

        def compute(self):
            return self.x

    a = A()
    assert a.x == a.compute()
    a.update(1)
    assert a.x == a.compute()
    a.update(2)
    assert a.x == a.compute()
