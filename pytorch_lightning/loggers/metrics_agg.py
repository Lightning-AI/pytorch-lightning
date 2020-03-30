import functools
import operator
from typing import Dict, Callable, Mapping, Sequence


def merge_dicts(dicts: Sequence[Mapping], fn: Callable[[Sequence[float]], float]) -> Dict:
    """Merges two dictionaries values with the given function.

    Args:
        dicts (list of dicts):
            Sequence of positional dict arguments to be merged.
        fn:
            Function which will be applied to the sequence of values from the same key of all dicts.
            If some dict has no required key, the value from this dict-key will not
            appear in the sequence to be aggregated.

    Returns (dict):
        Dictionary with merged values.

    Examples:
        >>> import pprint
        >>> d1 = {'a': 1.7, 'b': 2.0, 'c': 1}
        >>> d2 = {'a': 1.1, 'b': 2.2}
        >>> d3 = {'a': 1.1, 'v': 2.3}
        >>> fn = max
        >>> pprint.pprint(merge_dicts([d1, d2, d3], fn))
        {'a': 1.7, 'b': 2.2, 'c': 1, 'v': 2.3}
    """

    keys = list(functools.reduce(operator.or_, [set(d.keys()) for d in dicts]))
    dx = {k: fn([v for v in [d.get(k) for d in dicts] if v is not None]) for k in keys}
    return dx


def metrics_agg_sum(metrics_to_agg: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregates metric dictionaries sequences with sum function.

    Args:
        metrics_to_agg (Sequence[MetricsT]):
            Sequence with metrics dictionaries to be aggregated.

    Examples:
        >>> import pprint
        >>> metrics_to_agg = [{'a': 1.7, 'b': 2.0}, {'a': 1.1, 'b': 2.2}, {'a': 0.0, 'b': 0.1, 'c': 0.2}]
        >>> pprint.pprint(metrics_agg_sum(metrics_to_agg))
        {'a': 2.8, 'b': 4.3, 'c': 0.2}
    """
    return merge_dicts(metrics_to_agg, sum)


def metrics_agg_max(metrics_to_agg: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregates metric dictionaries sequences with max function.

    Args:
        metrics_to_agg (Sequence[MetricsT]):
            Sequence with metrics dictionaries to be aggregated.

    Examples:
        >>> import pprint
        >>> metrics_to_agg = [{'a': 1.7, 'b': 2.0}, {'a': 1.1, 'd': 2.2}, {'b': 0.0, 'c': 0.1}]
        >>> pprint.pprint(metrics_agg_max(metrics_to_agg))
        {'a': 1.7, 'b': 2.0, 'c': 0.1, 'd': 2.2}
    """
    return merge_dicts(metrics_to_agg, max)


def metrics_agg_min(metrics_to_agg: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Aggregates metric dictionaries sequences with min function.

    Args:
        metrics_to_agg (Sequence[MetricsT]):
            Sequence with metrics dictionaries to be aggregated.

    Examples:
        >>> import pprint
        >>> metrics_to_agg = [{'a': 1.7, 'b': 2.0}, {'a': 1.1, 'b': 2.2}, {'a': 0.0, 'b': 0.1}]
        >>> pprint.pprint(metrics_agg_min(metrics_to_agg))
        {'a': 0.0, 'b': 0.1}
    """
    return merge_dicts(metrics_to_agg, min)


def metrics_agg_avg(metrics_to_agg: Sequence[Dict[str, float]]):
    """Aggregates metric dictionaries sequences with average function.

    Args:
        metrics_to_agg (Sequence[MetricsT]):
            Sequence with metrics dictionaries to be aggregated.

    Examples:
        >>> import pprint
        >>> metrics_to_agg = [{'a': 1.9, 'b': 2.2}, {'a': 1.1, 'b': 2.2}, {'a': 0.0, 'b': 0.1, 'c': 3}]
        >>> pprint.pprint(metrics_agg_avg(metrics_to_agg))
        {'a': 1.0, 'b': 1.5, 'c': 1.0}
    """
    agg_mets = metrics_agg_sum(metrics_to_agg)
    agg_mets = {k: v / len(metrics_to_agg) for k, v in agg_mets.items()}
    return agg_mets
