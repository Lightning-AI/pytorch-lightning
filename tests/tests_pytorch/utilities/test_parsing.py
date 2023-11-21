# Copyright The Lightning AI team.
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
import inspect
import threading

import pytest
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.utilities.parsing import (
    _get_init_args,
    clean_namespace,
    collect_init_args,
    is_picklable,
    lightning_getattr,
    lightning_hasattr,
    lightning_setattr,
    parse_class_init_keys,
)
from torch.jit import ScriptModule

unpicklable_function = lambda: None


def model_and_trainer_cases():
    class TestHparamsNamespace(LightningModule):
        learning_rate = 1

        def __contains__(self, item):
            return item == "learning_rate"

    TestHparamsDict = {"learning_rate": 2}

    class TestModel1(LightningModule):  # test for namespace
        learning_rate = 0

    model1 = TestModel1()

    class TestModel2(LightningModule):  # test for hparams namespace
        hparams = TestHparamsNamespace()

    model2 = TestModel2()

    class TestModel3(LightningModule):  # test for hparams dict
        hparams = TestHparamsDict

    model3 = TestModel3()

    class TestModel4(LightningModule):  # fail case
        batch_size = 1

    model4 = TestModel4()
    trainer1 = Trainer()
    model4.trainer = trainer1
    datamodule = LightningDataModule()
    datamodule.batch_size = 8
    trainer1.datamodule = datamodule

    model5 = LightningModule()
    model5.trainer = trainer1

    class TestModel6(LightningModule):  # test for datamodule w/ hparams w/o attribute (should use datamodule)
        hparams = TestHparamsDict

    model6 = TestModel6()
    model6.trainer = trainer1

    TestHparamsDict2 = {"batch_size": 2}

    class TestModel7(LightningModule):  # test for datamodule w/ hparams w/ attribute (should use datamodule)
        hparams = TestHparamsDict2

    model7 = TestModel7()
    model7.trainer = trainer1

    class TestDataModule8(LightningDataModule):  # test for hparams dict
        hparams = TestHparamsDict2

    model8 = TestModel1()
    trainer2 = Trainer()
    model8.trainer = trainer2
    datamodule = TestDataModule8()
    trainer2.datamodule = datamodule

    return (model1, model2, model3, model4, model5, model6, model7, model8), (trainer1, trainer2)


def test_lightning_hasattr():
    """Test that the lightning_hasattr works in all cases."""
    models, _ = model_and_trainer_cases()
    model1, model2, model3, model4, model5, model6, model7, model8 = models
    assert lightning_hasattr(model1, "learning_rate"), "lightning_hasattr failed to find namespace variable"
    assert lightning_hasattr(model2, "learning_rate"), "lightning_hasattr failed to find hparams namespace variable"
    assert lightning_hasattr(model3, "learning_rate"), "lightning_hasattr failed to find hparams dict variable"
    assert not lightning_hasattr(model4, "learning_rate"), "lightning_hasattr found variable when it should not"
    assert lightning_hasattr(model5, "batch_size"), "lightning_hasattr failed to find batch_size in datamodule"
    assert lightning_hasattr(
        model6, "batch_size"
    ), "lightning_hasattr failed to find batch_size in datamodule w/ hparams present"
    assert lightning_hasattr(
        model7, "batch_size"
    ), "lightning_hasattr failed to find batch_size in hparams w/ datamodule present"
    assert lightning_hasattr(model8, "batch_size")

    for m in models:
        assert not lightning_hasattr(m, "this_attr_not_exist")


def test_lightning_getattr():
    """Test that the lightning_getattr works in all cases."""
    models, _ = model_and_trainer_cases()
    *__, model5, model6, model7, model8 = models
    for i, m in enumerate(models[:3]):
        value = lightning_getattr(m, "learning_rate")
        assert value == i, "attribute not correctly extracted"

    assert lightning_getattr(model5, "batch_size") == 8, "batch_size not correctly extracted"
    assert lightning_getattr(model6, "batch_size") == 8, "batch_size not correctly extracted"
    assert lightning_getattr(model7, "batch_size") == 8, "batch_size not correctly extracted"
    assert lightning_getattr(model8, "batch_size") == 2, "batch_size not correctly extracted"

    for m in models:
        with pytest.raises(
            AttributeError,
            match="is neither stored in the model namespace nor the `hparams` namespace/dict, nor the datamodule.",
        ):
            lightning_getattr(m, "this_attr_not_exist")


def test_lightning_setattr():
    """Test that the lightning_setattr works in all cases."""
    models, _ = model_and_trainer_cases()
    *__, model5, model6, model7, model8 = models
    for m in models[:3]:
        lightning_setattr(m, "learning_rate", 10)
        assert lightning_getattr(m, "learning_rate") == 10, "attribute not correctly set"

    lightning_setattr(model5, "batch_size", 128)
    lightning_setattr(model6, "batch_size", 128)
    lightning_setattr(model7, "batch_size", 128)
    assert lightning_getattr(model5, "batch_size") == 128, "batch_size not correctly set"
    assert lightning_getattr(model6, "batch_size") == 128, "batch_size not correctly set"
    assert lightning_getattr(model7, "batch_size") == 128, "batch_size not correctly set"
    assert lightning_getattr(model8, "batch_size") == 128, "batch_size not correctly set"

    for m in models:
        with pytest.raises(
            AttributeError,
            match="is neither stored in the model namespace nor the `hparams` namespace/dict, nor the datamodule.",
        ):
            lightning_setattr(m, "this_attr_not_exist", None)


def test_is_picklable():
    # See the full list of picklable types at
    # https://docs.python.org/3/library/pickle.html#pickle-picklable
    class UnpicklableClass:
        # Only classes defined at the top level of a module are picklable.
        pass

    true_cases = [None, True, 123, "str", (123, "str"), max]
    false_cases = [unpicklable_function, UnpicklableClass, ScriptModule(), threading.Lock()]

    for case in true_cases:
        assert is_picklable(case) is True

    for case in false_cases:
        assert is_picklable(case) is False


def test_clean_namespace():
    # See the full list of picklable types at
    # https://docs.python.org/3/library/pickle.html#pickle-picklable
    class UnpicklableClass:
        # Only classes defined at the top level of a module are picklable.
        pass

    test_case = {"1": None, "2": True, "3": 123, "4": unpicklable_function, "5": UnpicklableClass}

    clean_namespace(test_case)

    assert test_case == {"1": None, "2": True, "3": 123}


def test_parse_class_init_keys():
    class Class:
        def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
            pass

    assert parse_class_init_keys(Class) == ("self", "my_args", "my_kwargs")


def test_get_init_args():
    class AutomaticArgsModel:
        def __init__(self, anyarg, anykw=42, **kwargs):
            super().__init__()

            self.get_init_args_wrapper()

        def get_init_args_wrapper(self):
            frame = inspect.currentframe().f_back
            self.result = _get_init_args(frame)

    my_class = AutomaticArgsModel("test", anykw=32, otherkw=123)
    assert my_class.result == (my_class, {"anyarg": "test", "anykw": 32, "otherkw": 123})

    my_class.get_init_args_wrapper()
    assert my_class.result == (None, {})


def test_collect_init_args():
    class AutomaticArgsParent:
        def __init__(self, anyarg, anykw=42, **kwargs):
            super().__init__()
            self.get_init_args_wrapper()

        def get_init_args_wrapper(self):
            frame = inspect.currentframe()
            self.result = collect_init_args(frame, [])

    class AutomaticArgsChild(AutomaticArgsParent):
        def __init__(self, anyarg, childarg, anykw=42, childkw=42, **kwargs):
            super().__init__(anyarg, anykw=anykw, **kwargs)

    my_class = AutomaticArgsChild("test1", "test2", anykw=32, childkw=22, otherkw=123)
    assert my_class.result[0] == {"anyarg": "test1", "anykw": 32, "otherkw": 123}
    assert my_class.result[1] == {"anyarg": "test1", "childarg": "test2", "anykw": 32, "childkw": 22, "otherkw": 123}
