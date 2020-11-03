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
import pytest
import pickle

from pytorch_lightning.utilities.parsing import lightning_getattr
from pytorch_lightning.utilities.parsing import lightning_hasattr
from pytorch_lightning.utilities.parsing import lightning_setattr
from pytorch_lightning.utilities.parsing import str_to_bool_or_str
from pytorch_lightning.utilities.parsing import str_to_bool
from pytorch_lightning.utilities.parsing import is_picklable
from pytorch_lightning.utilities.parsing import clean_namespace
from pytorch_lightning.utilities.parsing import AttributeDict


def _get_test_cases():
    class TestHparamsNamespace:
        learning_rate = 1

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

    return model1, model2, model3, model4


def test_lightning_hasattr(tmpdir):
    """ Test that the lightning_hasattr works in all cases"""
    model1, model2, model3, model4 = _get_test_cases()
    assert lightning_hasattr(model1, 'learning_rate'), \
        'lightning_hasattr failed to find namespace variable'
    assert lightning_hasattr(model2, 'learning_rate'), \
        'lightning_hasattr failed to find hparams namespace variable'
    assert lightning_hasattr(model3, 'learning_rate'), \
        'lightning_hasattr failed to find hparams dict variable'
    assert not lightning_hasattr(model4, 'learning_rate'), \
        'lightning_hasattr found variable when it should not'


def test_lightning_getattr(tmpdir):
    """ Test that the lightning_getattr works in all cases"""
    models = _get_test_cases()
    for i, m in enumerate(models[:3]):
        value = lightning_getattr(m, 'learning_rate')
        assert value == i, 'attribute not correctly extracted'


def test_lightning_setattr(tmpdir):
    """ Test that the lightning_setattr works in all cases"""
    models = _get_test_cases()
    for m in models[:3]:
        lightning_setattr(m, 'learning_rate', 10)
        assert lightning_getattr(m, 'learning_rate') == 10, \
            'attribute not correctly set'


def test_str_to_bool_or_str(tmpdir):
    true_cases = ['y', 'yes', 't', 'true', 'on', '1']
    false_cases = ['n', 'no', 'f', 'false', 'off', '0']
    other_cases = ['yyeess', 'noooo', 'lightning']

    for case in true_cases:
        assert str_to_bool_or_str(case) is True

    for case in false_cases:
        assert str_to_bool_or_str(case) is False

    for case in other_cases:
        assert str_to_bool_or_str(case) == case


def test_str_to_bool(tmpdir):
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


def test_is_picklable(tmpdir):
    # See the full list of picklable types at
    # https://docs.python.org/3/library/pickle.html#pickle-picklable
    class UnpicklableClass:
        # Only classes defined at the top level of a module are picklable.
        pass

    def unpicklable_function():
        # Only functions defined at the top level of a module are picklable.
        pass
    
    true_cases = [None, True, 123, "str", (123, "str"), max]
    false_cases = [unpicklable_function, UnpicklableClass]

    for case in true_cases:
        assert is_picklable(case) is True
        
    for case in false_cases:
        assert is_picklable(case) is False


def test_clean_namespace(tmpdir):
    class UnpicklableClass:
        pass

    def unpicklable_function():
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


def test_AttributeDict(tmpdir):
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
