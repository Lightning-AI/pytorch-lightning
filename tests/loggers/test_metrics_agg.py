import operator

import pytest

from pytorch_lightning.loggers import metrics_agg


@pytest.mark.parametrize(
    'd1,d2,fn,res', [
        ({'a': 1, 'b': 2}, {'a': -1, 'b': -2}, operator.add, {'a': 0, 'b': 0}),
        ({'a': 0, 'b': 2}, {'a': 2, 'b': 0}, max, {'a': 2, 'b': 2}),
        ({'a': 0, 'b': 2}, {'a': 2, 'b': 0}, min, {'a': 0, 'b': 0})
    ]
)
def test_merge_two_dicts(d1, d2, fn, res):
    _res = metrics_agg.merge_two_dicts(d1, d2, fn)
    assert _res == res


@pytest.mark.parametrize(
    'metrics,fn,res', [
        ([{'a': 1, 'b': 2}, {'a': -1, 'b': -2}, {'a': 1, 'b': 1}], operator.add, {'a': 1, 'b': 1}),
        ([{'a': 0, 'b': 2}, {'a': 2, 'b': 0}], max, {'a': 2, 'b': 2}),
        ([{'a': 0, 'b': 2}], min, {'a': 0, 'b': 2})
    ]
)
def test_metrics_agg_simple(metrics, fn, res):
    _res = metrics_agg.metrics_agg_simple(metrics, fn)
    assert _res == res


@pytest.mark.parametrize(
    'metrics,res', [
        ([{'a': 1, 'b': 2}, {'a': -1, 'b': -2}, {'a': 1, 'b': 1}], {'a': 1, 'b': 1}),
        ([{'a': 0, 'b': 2}, {'a': 2, 'b': 0}], {'a': 2, 'b': 2}),
        ([{'a': 0, 'b': 2}], {'a': 0, 'b': 2})
    ]
)
def test_metrics_agg_sum(metrics, res):
    _res = metrics_agg.metrics_agg_sum(metrics)
    assert _res == res


@pytest.mark.parametrize(
    'metrics,res', [
        ([{'a': 1, 'b': 2}, {'a': -1, 'b': -2}, {'a': 1, 'b': 1}], {'a': -1, 'b': -2}),
        ([{'a': 0, 'b': 2}, {'a': 2, 'b': 0}], {'a': 0, 'b': 0}),
        ([{'a': 0, 'b': 2}], {'a': 0, 'b': 2})
    ]
)
def test_metrics_agg_min(metrics, res):
    _res = metrics_agg.metrics_agg_min(metrics)
    assert _res == res


@pytest.mark.parametrize(
    'metrics,res', [
        ([{'a': 1, 'b': 2}, {'a': -1, 'b': -2}, {'a': 1, 'b': 1}], {'a': 1, 'b': 2}),
        ([{'a': 0, 'b': 2}, {'a': 2, 'b': 0}], {'a': 2, 'b': 2}),
        ([{'a': 0, 'b': 2}], {'a': 0, 'b': 2})
    ]
)
def test_metrics_agg_max(metrics, res):
    _res = metrics_agg.metrics_agg_max(metrics)
    assert _res == res


@pytest.mark.parametrize(
    'metrics,res', [
        ([{'a': 0, 'b': 0}, {'a': 1, 'b': 1}, {'a': 2, 'b': 2}], {'a': 1, 'b': 1}),
        ([{'a': 1, 'b': 2}, {'a': 1, 'b': 2}], {'a': 1, 'b': 2}),
        ([{'a': 0, 'b': 2}], {'a': 0, 'b': 2})
    ]
)
def test_metrics_agg_avg(metrics, res):
    _res = metrics_agg.metrics_agg_avg(metrics)
    assert _res == res
