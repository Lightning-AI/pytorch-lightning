from contextlib import contextmanager
from functools import partial, wraps
from typing import Any, Callable
from unittest.mock import Mock

import pytest

from lightning_utilities.core.overrides import is_overridden


class LightningModule:
    def training_step(self): ...


class BoringModel(LightningModule):
    def training_step(self): ...


class Strategy:
    @contextmanager
    def model_sharded_context(): ...


class SingleDeviceStrategy(Strategy): ...


def test_is_overridden():
    assert not is_overridden("whatever", object(), parent=LightningModule)

    class TestModel(BoringModel):
        def foo(self):
            pass

        def bar(self):
            return 1

    with pytest.raises(ValueError, match="The parent should define the method"):
        is_overridden("foo", TestModel(), parent=BoringModel)

    # normal usage
    assert is_overridden("training_step", BoringModel(), parent=LightningModule)

    # reversed. works even without inheritance
    assert is_overridden("training_step", LightningModule(), parent=BoringModel)

    class WrappedModel(TestModel):
        def __new__(cls, *args: Any, **kwargs: Any):
            obj = super().__new__(cls)
            obj.foo = cls.wrap(obj.foo)
            obj.bar = cls.wrap(obj.bar)
            return obj

        @staticmethod
        def wrap(fn) -> Callable:
            @wraps(fn)
            def wrapper():
                fn()

            return wrapper

        def bar(self):
            return 2

    # `functools.wraps()` support
    assert not is_overridden("foo", WrappedModel(), parent=TestModel)
    assert is_overridden("bar", WrappedModel(), parent=TestModel)

    # `Mock` support
    mock = Mock(spec=BoringModel, wraps=BoringModel())
    assert is_overridden("training_step", mock, parent=LightningModule)

    # `partial` support
    model = BoringModel()
    model.training_step = partial(model.training_step)
    assert is_overridden("training_step", model, parent=LightningModule)

    # `@contextmanager` support
    assert not is_overridden("model_sharded_context", SingleDeviceStrategy(), Strategy)
