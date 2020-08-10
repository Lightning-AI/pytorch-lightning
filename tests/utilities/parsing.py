import pytest

from pytorch_lightning.utilities.parsing import lightning_getattr, lightning_hasattr, lightning_setattr


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
