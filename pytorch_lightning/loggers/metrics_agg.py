import functools
import operator
from typing import Dict, Callable, Mapping, Sequence

MetricsT = Dict[str, float]
MetricsAggFnT = Callable[[Sequence[MetricsT]], MetricsT]


def merge_two_dicts(d1: Mapping, d2: Mapping, fn: Callable[[float, float], float], default_value: float = 0) -> Dict:
    """Merges two dictionaries values with the given function.

    Args:
        d1 (dict):
            First dictionary
        d2 (dict):
            Second dictionary
        fn:
            Function which will be applied to two values from the same key of both dicts.
        default_value (float):
            If value is presented only in one dict, it will be aggregated with `default_value`.

    Returns (dict):
        Dictionary with merged values.

    Examples:
        >>> import pprint
        >>> d1 = {'a': 1.7, 'b': 2.0, 'c': 1}
        >>> d2 = {'a': 1.1, 'b': 2.2}
        >>> fn = max
        >>> pprint.pprint(merge_two_dicts(d1, d2, fn))
        {'a': 1.7, 'b': 2.2, 'c': 1}
    """

    keys = set(list(d1.keys()) + list(d2.keys()))
    dx = {k: fn([v for v in (d1.get(k), d2.get(k)) if v]) for k in keys}
    return dx


def metrics_agg_simple(metrics_to_agg: Sequence[MetricsT], fn: Callable[[float, float], float]) -> MetricsT:
    """Aggregates metrics dictionaries with the given function.

    Args:
        metrics_to_agg (Sequence[MetricsT]):
            Sequence with metrics dictionaries to be aggregated.
        fn: Values reduction function (check `merge_two_dicts`).

    Returns (MetricsT):
        Aggregated metrics dictionary.

    Examples:
        >>> import pprint
        >>> import operator
        >>> metrics_to_agg = [{'a': 1.7, 'b': 2.0}, {'a': 1.1, 'b': 2.2}, {'a': 0.0, 'b': 0.1}]
        >>> pprint.pprint(metrics_agg_simple(metrics_to_agg, operator.add))
        {'a': 2.8, 'b': 4.3}
    """
    return functools.reduce(lambda d1, d2: merge_two_dicts(d1, d2, fn), metrics_to_agg)


def metrics_agg_sum(metrics_to_agg: Sequence[MetricsT]) -> MetricsT:
    """Aggregates metric dictionaries sequences with sum function.

    Args:
        Check `metrics_agg_simple` function for args and return description.

    Examples:
        >>> import pprint
        >>> metrics_to_agg = [{'a': 1.7, 'b': 2.0}, {'a': 1.1, 'b': 2.2}, {'a': 0.0, 'b': 0.1}]
        >>> pprint.pprint(metrics_agg_sum(metrics_to_agg))
        {'a': 2.8, 'b': 4.3}
    """
    return metrics_agg_simple(metrics_to_agg, operator.add)


def metrics_agg_max(metrics_to_agg: Sequence[MetricsT]) -> MetricsT:
    """Aggregates metric dictionaries sequences with max function.

    Args:
        Check `metrics_agg_simple` function for args and return description.

    Examples:
        >>> import pprint
        >>> metrics_to_agg = [{'a': 1.7, 'b': 2.0}, {'a': 1.1, 'b': 2.2}, {'a': 0.0, 'b': 0.1}]
        >>> pprint.pprint(metrics_agg_max(metrics_to_agg))
        {'a': 1.7, 'b': 2.2}
    """
    return metrics_agg_simple(metrics_to_agg, max)


def metrics_agg_min(metrics_to_agg: Sequence[MetricsT]) -> MetricsT:
    """Aggregates metric dictionaries sequences with min function.

    Args:
        Check `metrics_agg_simple` function for args and return description.

    Examples:
        >>> import pprint
        >>> metrics_to_agg = [{'a': 1.7, 'b': 2.0}, {'a': 1.1, 'b': 2.2}, {'a': 0.0, 'b': 0.1}]
        >>> pprint.pprint(metrics_agg_min(metrics_to_agg))
        {'a': 0.0, 'b': 0.1}
    """
    return metrics_agg_simple(metrics_to_agg, min)


def metrics_agg_avg(metrics_to_agg: Sequence[MetricsT]):
    """Aggregates metric dictionaries sequences with average function.

    Args:
        Check `metrics_agg_simple` function for args and return description.

    Examples:
        >>> import pprint
        >>> metrics_to_agg = [{'a': 1.9, 'b': 2.2}, {'a': 1.1, 'b': 2.2}, {'a': 0.0, 'b': 0.1}]
        >>> pprint.pprint(metrics_agg_avg(metrics_to_agg))
        {'a': 1.0, 'b': 1.5}
    """
    agg_mets = metrics_agg_sum(metrics_to_agg)
    agg_mets = {k: v / len(metrics_to_agg) for k, v in agg_mets.items()}
    return agg_mets
