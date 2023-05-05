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

from argparse import Namespace
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import numpy as np
from torch import Tensor


def _convert_params(params: Optional[Union[Dict[str, Any], Namespace]]) -> Dict[str, Any]:
    """Ensure parameters are a dict or convert to dict if necessary.

    Args:
        params: Target to be converted to a dictionary

    Returns:
        params as a dictionary
    """
    # in case converting from namespace
    if isinstance(params, Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params


def _sanitize_callable_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize callable params dict, e.g. ``{'a': <function_**** at 0x****>} -> {'a': 'function_****'}``.

    Args:
        params: Dictionary containing the hyperparameters

    Returns:
        dictionary with all callables sanitized
    """

    def _sanitize_callable(val: Any) -> Any:
        # Give them one chance to return a value. Don't go rabbit hole of recursive call
        if callable(val):
            try:
                _val = val()
                if callable(_val):
                    return val.__name__
                return _val
            # todo: specify the possible exception
            except Exception:
                return getattr(val, "__name__", None)
        return val

    return {key: _sanitize_callable(val) for key, val in params.items()}


def _flatten_dict(params: MutableMapping[Any, Any], delimiter: str = "/", parent_key: str = "") -> Dict[str, Any]:
    """Flatten hierarchical dict, e.g. ``{'a': {'b': 'c'}} -> {'a/b': 'c'}``.

    Args:
        params: Dictionary containing the hyperparameters
        delimiter: Delimiter to express the hierarchy. Defaults to ``'/'``.

    Returns:
        Flattened dict.

    Examples:
        >>> _flatten_dict({'a': {'b': 'c'}})
        {'a/b': 'c'}
        >>> _flatten_dict({'a': {'b': 123}})
        {'a/b': 123}
        >>> _flatten_dict({5: {'a': 123}})
        {'5/a': 123}
    """
    result: Dict[str, Any] = {}
    for k, v in params.items():
        new_key = parent_key + delimiter + str(k) if parent_key else str(k)
        if isinstance(v, Namespace):
            v = vars(v)
        if isinstance(v, MutableMapping):
            result = {**result, **_flatten_dict(v, parent_key=new_key, delimiter=delimiter)}
        else:
            result[new_key] = v
    return result


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Returns params with non-primitvies converted to strings for logging.

    >>> import torch
    >>> params = {"float": 0.3,
    ...           "int": 1,
    ...           "string": "abc",
    ...           "bool": True,
    ...           "list": [1, 2, 3],
    ...           "namespace": Namespace(foo=3),
    ...           "layer": torch.nn.BatchNorm1d}
    >>> import pprint
    >>> pprint.pprint(_sanitize_params(params))  # doctest: +NORMALIZE_WHITESPACE
    {'bool': True,
        'float': 0.3,
        'int': 1,
        'layer': "<class 'torch.nn.modules.batchnorm.BatchNorm1d'>",
        'list': '[1, 2, 3]',
        'namespace': 'Namespace(foo=3)',
        'string': 'abc'}
    """
    for k in params:
        # convert relevant np scalars to python types first (instead of str)
        if isinstance(params[k], (np.bool_, np.integer, np.floating)):
            params[k] = params[k].item()
        elif type(params[k]) not in [bool, int, float, str, Tensor]:
            params[k] = str(params[k])
    return params


def _add_prefix(
    metrics: Mapping[str, Union[Tensor, float]], prefix: str, separator: str
) -> Mapping[str, Union[Tensor, float]]:
    """Insert prefix before each key in a dict, separated by the separator.

    Args:
        metrics: Dictionary with metric names as keys and measured quantities as values
        prefix: Prefix to insert before each key
        separator: Separates prefix and original key name

    Returns:
        Dictionary with prefix and separator inserted before each key
    """
    if not prefix:
        return metrics
    return {f"{prefix}{separator}{k}": v for k, v in metrics.items()}
