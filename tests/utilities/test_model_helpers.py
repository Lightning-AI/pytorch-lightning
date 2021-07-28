# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from functools import partial, wraps
from unittest.mock import Mock

import pytest

from pytorch_lightning import LightningDataModule, Trainer
from pytorch_lightning.utilities.model_helpers import is_overridden
from tests.helpers import BoringDataModule, BoringModel


def test_is_overridden():
    model = BoringModel()
    datamodule = BoringDataModule()

    # edge cases
    assert not is_overridden("whatever", None)
    with pytest.raises(ValueError, match="Expected a parent"):
        is_overridden("whatever", object())
    assert not is_overridden("whatever", model)
    assert not is_overridden("whatever", model, parent=LightningDataModule)

    class TestModel(BoringModel):
        def foo(self):
            pass

        def bar(self):
            return 1

    with pytest.raises(ValueError, match="The parent should define the method"):
        is_overridden("foo", TestModel())

    # normal usage
    assert is_overridden("training_step", model)
    assert is_overridden("train_dataloader", datamodule)

    class WrappedModel(TestModel):
        def __new__(cls, *args, **kwargs):
            obj = super().__new__(cls)
            obj.foo = cls.wrap(obj.foo)
            obj.bar = cls.wrap(obj.bar)
            return obj

        @staticmethod
        def wrap(fn):
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
    mock = Mock(spec=BoringModel, wraps=model)
    assert is_overridden("training_step", mock)
    mock = Mock(spec=BoringDataModule, wraps=datamodule)
    assert is_overridden("train_dataloader", mock)

    # `partial` support
    model.training_step = partial(model.training_step)
    assert is_overridden("training_step", model)

    # `_PatchDataLoader.patch_loader_code` support
    class TestModel(BoringModel):
        def on_fit_start(self):
            assert is_overridden("train_dataloader", self)
            self.on_fit_start_called = True

    model = TestModel()
    trainer = Trainer(fast_dev_run=1)
    trainer.fit(model, train_dataloader=model.train_dataloader())
    assert model.on_fit_start_called
