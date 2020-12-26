import pickle
from collections import OrderedDict
from distutils.version import LooseVersion

import cloudpickle
import numpy as np
import pytest
import torch

from pytorch_lightning.metrics.metric import Metric

torch.manual_seed(42)


class Dummy(Metric):
    name = "Dummy"

    def __init__(self):
        super().__init__()
        self.add_state("x", torch.tensor(0), dist_reduce_fx=None)

    def update(self):
        pass

    def compute(self):
        pass


class DummyList(Metric):
    name = "DummyList"

    def __init__(self):
        super().__init__()
        self.add_state("x", list(), dist_reduce_fx=None)

    def update(self):
        pass

    def compute(self):
        pass


def test_inherit():
    a = Dummy()


def test_add_state():
    a = Dummy()

    a.add_state("a", torch.tensor(0), "sum")
    assert a._reductions["a"](torch.tensor([1, 1])) == 2

    a.add_state("b", torch.tensor(0), "mean")
    assert np.allclose(a._reductions["b"](torch.tensor([1.0, 2.0])).numpy(), 1.5)

    a.add_state("c", torch.tensor(0), "cat")
    assert a._reductions["c"]([torch.tensor([1]), torch.tensor([1])]).shape == (2,)

    with pytest.raises(ValueError):
        a.add_state("d1", torch.tensor(0), 'xyz')

    with pytest.raises(ValueError):
        a.add_state("d2", torch.tensor(0), 42)

    with pytest.raises(ValueError):
        a.add_state("d3", [torch.tensor(0)], 'sum')

    with pytest.raises(ValueError):
        a.add_state("d4", 42, 'sum')

    def custom_fx(x):
        return -1

    a.add_state("e", torch.tensor(0), custom_fx)
    assert a._reductions["e"](torch.tensor([1, 1])) == -1


def test_add_state_persistent():
    a = Dummy()

    a.add_state("a", torch.tensor(0), "sum", persistent=True)
    assert "a" in a.state_dict()

    a.add_state("b", torch.tensor(0), "sum", persistent=False)

    if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
        assert "b" not in a.state_dict()


def test_reset():
    class A(Dummy):
        pass

    class B(DummyList):
        pass

    a = A()
    assert a.x == 0
    a.x = torch.tensor(5)
    a.reset()
    assert a.x == 0

    b = B()
    assert isinstance(b.x, list) and len(b.x) == 0
    b.x = torch.tensor(5)
    b.reset()
    assert isinstance(b.x, list) and len(b.x) == 0


def test_update():
    class A(Dummy):
        def update(self, x):
            self.x += x

    a = A()
    assert a.x == 0
    assert a._computed is None
    a.update(1)
    assert a._computed is None
    assert a.x == 1
    a.update(2)
    assert a.x == 3
    assert a._computed is None


def test_compute():
    class A(Dummy):
        def update(self, x):
            self.x += x

        def compute(self):
            return self.x

    a = A()
    assert 0 == a.compute()
    assert 0 == a.x
    a.update(1)
    assert a._computed is None
    assert a.compute() == 1
    assert a._computed == 1
    a.update(2)
    assert a._computed is None
    assert a.compute() == 2
    assert a._computed == 2

    # called without update, should return cached value
    a._computed = 5
    assert a.compute() == 5


def test_forward():
    class A(Dummy):
        def update(self, x):
            self.x += x

        def compute(self):
            return self.x

    a = A()
    assert a(5) == 5
    assert a._forward_cache == 5

    assert a(8) == 8
    assert a._forward_cache == 8

    assert a.compute() == 13


class ToPickle(Dummy):
    def update(self, x):
        self.x += x

    def compute(self):
        return self.x


def test_pickle(tmpdir):
    # doesn't tests for DDP
    a = ToPickle()
    a.update(1)

    metric_pickled = pickle.dumps(a)
    metric_loaded = pickle.loads(metric_pickled)

    assert metric_loaded.compute() == 1

    metric_loaded.update(5)
    assert metric_loaded.compute() == 5

    metric_pickled = cloudpickle.dumps(a)
    metric_loaded = cloudpickle.loads(metric_pickled)

    assert metric_loaded.compute() == 1


def test_state_dict(tmpdir):
    """ test that metric states can be removed and added to state dict """
    metric = Dummy()
    assert metric.state_dict() == OrderedDict()
    metric.persistent(True)
    assert metric.state_dict() == OrderedDict(x=0)
    metric.persistent(False)
    assert metric.state_dict() == OrderedDict()
