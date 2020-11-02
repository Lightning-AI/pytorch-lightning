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

from pytorch_lightning.utilities.parsing import lightning_getattr
from pytorch_lightning.utilities.parsing import lightning_hasattr
from pytorch_lightning.utilities.parsing import lightning_setattr
from pytorch_lightning.utilities.parsing import str_to_bool_or_str
from pytorch_lightning.utilities.parsing import str_to_bool


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
