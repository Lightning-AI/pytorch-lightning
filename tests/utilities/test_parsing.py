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
import inspect

import pytest
from torch.jit import ScriptModule

from pytorch_lightning.utilities.parsing import (
    AttributeDict,
    clean_namespace,
    collect_init_args,
    flatten_dict,
    get_init_args,
    is_picklable,
    lightning_getattr,
    lightning_hasattr,
    lightning_setattr,
    parse_class_init_keys,
    str_to_bool,
    str_to_bool_or_int,
    str_to_bool_or_str,
)

unpicklable_function = lambda: None


@pytest.fixture(scope="module")
def model_cases():

    class TestHparamsNamespace:
        learning_rate = 1

        def __contains__(self, item):
            return item == "learning_rate"

    TestHparamsDict = {'learning_rate': 2}

    class TestModel1:  # test for namespace
        learning_rate = 0

    model1 = TestModel1()

    class TestModel2:  # test for hparams namespace
        hparams = TestHparamsNamespace()

    model2 = TestModel2()

    class TestModel3:  # test for hparams dict
        hparams = TestHparamsDict

    model3 = TestModel3()

    class TestModel4:  # fail case
        batch_size = 1

    model4 = TestModel4()

    class DataModule:
        batch_size = 8

    class Trainer:
        datamodule = DataModule

    class TestModel5:  # test for datamodule
        trainer = Trainer

    model5 = TestModel5()

    class TestModel6:  # test for datamodule w/ hparams w/o attribute (should use datamodule)
        trainer = Trainer
        hparams = TestHparamsDict

    model6 = TestModel6()

    TestHparamsDict2 = {'batch_size': 2}

    class TestModel7:  # test for datamodule w/ hparams w/ attribute (should use datamodule)
        trainer = Trainer
        hparams = TestHparamsDict2

    model7 = TestModel7()

    return model1, model2, model3, model4, model5, model6, model7


def test_lightning_hasattr(tmpdir, model_cases):
    """Test that the lightning_hasattr works in all cases"""
    model1, model2, model3, model4, model5, model6, model7 = models = model_cases
    assert lightning_hasattr(model1, 'learning_rate'), \
        'lightning_hasattr failed to find namespace variable'
    assert lightning_hasattr(model2, 'learning_rate'), \
        'lightning_hasattr failed to find hparams namespace variable'
    assert lightning_hasattr(model3, 'learning_rate'), \
        'lightning_hasattr failed to find hparams dict variable'
    assert not lightning_hasattr(model4, 'learning_rate'), \
        'lightning_hasattr found variable when it should not'
    assert lightning_hasattr(model5, 'batch_size'), \
        'lightning_hasattr failed to find batch_size in datamodule'
    assert lightning_hasattr(model6, 'batch_size'), \
        'lightning_hasattr failed to find batch_size in datamodule w/ hparams present'
    assert lightning_hasattr(model7, 'batch_size'), \
        'lightning_hasattr failed to find batch_size in hparams w/ datamodule present'

    for m in models:
        assert not lightning_hasattr(m, "this_attr_not_exist")


def test_lightning_getattr(tmpdir, model_cases):
    """Test that the lightning_getattr works in all cases"""
    models = model_cases
    for i, m in enumerate(models[:3]):
        value = lightning_getattr(m, 'learning_rate')
        assert value == i, 'attribute not correctly extracted'

    model5, model6, model7 = models[4:]
    assert lightning_getattr(model5, 'batch_size') == 8, \
        'batch_size not correctly extracted'
    assert lightning_getattr(model6, 'batch_size') == 8, \
        'batch_size not correctly extracted'
    assert lightning_getattr(model7, 'batch_size') == 8, \
        'batch_size not correctly extracted'

    for m in models:
        with pytest.raises(
            AttributeError,
            match="is neither stored in the model namespace nor the `hparams` namespace/dict, nor the datamodule."
        ):
            lightning_getattr(m, "this_attr_not_exist")


def test_lightning_setattr(tmpdir, model_cases):
    """Test that the lightning_setattr works in all cases"""
    models = model_cases
    for m in models[:3]:
        lightning_setattr(m, 'learning_rate', 10)
        assert lightning_getattr(m, 'learning_rate') == 10, \
            'attribute not correctly set'

    model5, model6, model7 = models[4:]
    lightning_setattr(model5, 'batch_size', 128)
    lightning_setattr(model6, 'batch_size', 128)
    lightning_setattr(model7, 'batch_size', 128)
    assert lightning_getattr(model5, 'batch_size') == 128, \
        'batch_size not correctly set'
    assert lightning_getattr(model6, 'batch_size') == 128, \
        'batch_size not correctly set'
    assert lightning_getattr(model7, 'batch_size') == 128, \
        'batch_size not correctly set'

    for m in models:
        with pytest.raises(
            AttributeError,
            match="is neither stored in the model namespace nor the `hparams` namespace/dict, nor the datamodule."
        ):
            lightning_setattr(m, "this_attr_not_exist", None)


def test_str_to_bool_or_str():
    true_cases = ['y', 'yes', 't', 'true', 'on', '1']
    false_cases = ['n', 'no', 'f', 'false', 'off', '0']
    other_cases = ['yyeess', 'noooo', 'lightning']

    for case in true_cases:
        assert str_to_bool_or_str(case) is True

    for case in false_cases:
        assert str_to_bool_or_str(case) is False

    for case in other_cases:
        assert str_to_bool_or_str(case) == case


def test_str_to_bool():
    true_cases = ['y', 'yes', 't', 'true', 'on', '1']
    false_cases = ['n', 'no', 'f', 'false', 'off', '0']
    other_cases = ['yyeess', 'noooo', 'lightning']

    for case in true_cases:
        assert str_to_bool(case) is True

    for case in false_cases:
        assert str_to_bool(case) is False

    for case in other_cases:
        with pytest.raises(ValueError):
            str_to_bool(case)


def test_str_to_bool_or_int():
    assert str_to_bool_or_int("0") is False
    assert str_to_bool_or_int("1") is True
    assert str_to_bool_or_int("true") is True
    assert str_to_bool_or_int("2") == 2
    assert str_to_bool_or_int("abc") == "abc"


def test_is_picklable(tmpdir):
    # See the full list of picklable types at
    # https://docs.python.org/3/library/pickle.html#pickle-picklable
    class UnpicklableClass:
        # Only classes defined at the top level of a module are picklable.
        pass

    true_cases = [None, True, 123, "str", (123, "str"), max]
    false_cases = [unpicklable_function, UnpicklableClass, ScriptModule()]

    for case in true_cases:
        assert is_picklable(case) is True

    for case in false_cases:
        assert is_picklable(case) is False


def test_clean_namespace(tmpdir):
    # See the full list of picklable types at
    # https://docs.python.org/3/library/pickle.html#pickle-picklable
    class UnpicklableClass:
        # Only classes defined at the top level of a module are picklable.
        pass

    test_case = {
        "1": None,
        "2": True,
        "3": 123,
        "4": unpicklable_function,
        "5": UnpicklableClass,
    }

    clean_namespace(test_case)

    assert test_case == {"1": None, "2": True, "3": 123}


def test_parse_class_init_keys(tmpdir):

    class Class:

        def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
            pass

    assert parse_class_init_keys(Class) == ("self", "my_args", "my_kwargs")


def test_get_init_args(tmpdir):

    class AutomaticArgsModel:

        def __init__(self, anyarg, anykw=42, **kwargs):
            super().__init__()

            self.get_init_args_wrapper()

        def get_init_args_wrapper(self):
            frame = inspect.currentframe().f_back
            self.result = get_init_args(frame)

    my_class = AutomaticArgsModel("test", anykw=32, otherkw=123)
    assert my_class.result == {"anyarg": "test", "anykw": 32, "otherkw": 123}

    my_class.get_init_args_wrapper()
    assert my_class.result == {}


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


def test_attribute_dict(tmpdir):
    # Test initialization
    inputs = {
        'key1': 1,
        'key2': 'abc',
    }
    ad = AttributeDict(inputs)
    for key, value in inputs.items():
        assert getattr(ad, key) == value

    # Test adding new items
    ad = AttributeDict()
    ad.update({'key1': 1})
    assert ad.key1 == 1

    # Test updating existing items
    ad = AttributeDict({'key1': 1})
    ad.key1 = 123
    assert ad.key1 == 123


def test_flatten_dict(tmpdir):
    d = {'1': 1, '_': {'2': 2, '_': {'3': 3, '4': 4}}}

    expected = {
        '1': 1,
        '2': 2,
        '3': 3,
        '4': 4,
    }

    assert flatten_dict(d) == expected
