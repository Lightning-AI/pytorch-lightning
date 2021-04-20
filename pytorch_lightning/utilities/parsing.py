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
import copy
import inspect
import pickle
import types
from argparse import Namespace
from typing import Any, Dict, Optional, Sequence, Tuple, Union

from pytorch_lightning.utilities import rank_zero_warn


def str_to_bool_or_str(val: str) -> Union[str, bool]:
    """Possibly convert a string representation of truth to bool.
    Returns the input otherwise.
    Based on the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.
    """
    lower = val.lower()
    if lower in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif lower in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        return val


def str_to_bool(val: str) -> bool:
    """Convert a string representation of truth to bool.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.

    >>> str_to_bool('YES')
    True
    >>> str_to_bool('FALSE')
    False
    """
    val = str_to_bool_or_str(val)
    if isinstance(val, bool):
        return val
    raise ValueError(f'invalid truth value {val}')


def is_picklable(obj: object) -> bool:
    """Tests if an object can be pickled"""

    try:
        pickle.dumps(obj)
        return True
    except (pickle.PickleError, AttributeError):
        return False


def clean_namespace(hparams):
    """Removes all unpicklable entries from hparams"""

    hparams_dict = hparams
    if isinstance(hparams, Namespace):
        hparams_dict = hparams.__dict__

    del_attrs = [k for k, v in hparams_dict.items() if not is_picklable(v)]

    for k in del_attrs:
        rank_zero_warn(f"attribute '{k}' removed from hparams because it cannot be pickled", UserWarning)
        del hparams_dict[k]


def parse_class_init_keys(cls) -> Tuple[str, str, str]:
    """Parse key words for standard self, *args and **kwargs

    >>> class Model():
    ...     def __init__(self, hparams, *my_args, anykw=42, **my_kwargs):
    ...         pass
    >>> parse_class_init_keys(Model)
    ('self', 'my_args', 'my_kwargs')
    """
    init_parameters = inspect.signature(cls.__init__).parameters
    # docs claims the params are always ordered
    # https://docs.python.org/3/library/inspect.html#inspect.Signature.parameters
    init_params = list(init_parameters.values())
    # self is always first
    n_self = init_params[0].name

    def _get_first_if_any(params, param_type):
        for p in params:
            if p.kind == param_type:
                return p.name

    n_args = _get_first_if_any(init_params, inspect.Parameter.VAR_POSITIONAL)
    n_kwargs = _get_first_if_any(init_params, inspect.Parameter.VAR_KEYWORD)

    return n_self, n_args, n_kwargs


def get_init_args(frame) -> dict:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if '__class__' not in local_vars:
        return {}
    cls = local_vars['__class__']
    init_parameters = inspect.signature(cls.__init__).parameters
    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    filtered_vars = [n for n in (self_var, args_var, kwargs_var) if n]
    exclude_argnames = (*filtered_vars, '__class__', 'frame', 'frame_args')
    # only collect variables that appear in the signature
    local_args = {k: local_vars[k] for k in init_parameters.keys()}
    local_args.update(local_args.get(kwargs_var, {}))
    local_args = {k: v for k, v in local_args.items() if k not in exclude_argnames}
    return local_args


def collect_init_args(frame, path_args: list, inside: bool = False) -> list:
    """
    Recursively collects the arguments passed to the child constructors in the inheritance tree.

    Args:
        frame: the current stack frame
        path_args: a list of dictionaries containing the constructor args in all parent classes
        inside: track if we are inside inheritance path, avoid terminating too soon

    Return:
          A list of dictionaries where each dictionary contains the arguments passed to the
          constructor at that level. The last entry corresponds to the constructor call of the
          most specific class in the hierarchy.
    """
    _, _, _, local_vars = inspect.getargvalues(frame)
    if '__class__' in local_vars:
        local_args = get_init_args(frame)
        # recursive update
        path_args.append(local_args)
        return collect_init_args(frame.f_back, path_args, inside=True)
    elif not inside:
        return collect_init_args(frame.f_back, path_args, inside)
    else:
        return path_args


def flatten_dict(source, result=None):
    if result is None:
        result = {}

    for k, v in source.items():
        if isinstance(v, dict):
            _ = flatten_dict(v, result)
        else:
            result[k] = v

    return result


def save_hyperparameters(
    obj: Any,
    *args,
    ignore: Optional[Union[Sequence[str], str]] = None,
    frame: Optional[types.FrameType] = None
) -> None:
    """See :meth:`~pytorch_lightning.LightningModule.save_hyperparameters`"""
    if not frame:
        frame = inspect.currentframe().f_back
    init_args = get_init_args(frame)
    assert init_args, "failed to inspect the obj init"

    if ignore is not None:
        if isinstance(ignore, str):
            ignore = [ignore]
        if isinstance(ignore, (list, tuple)):
            ignore = [arg for arg in ignore if isinstance(arg, str)]
        init_args = {k: v for k, v in init_args.items() if k not in ignore}

    if not args:
        # take all arguments
        hp = init_args
        obj._hparams_name = "kwargs" if hp else None
    else:
        # take only listed arguments in `save_hparams`
        isx_non_str = [i for i, arg in enumerate(args) if not isinstance(arg, str)]
        if len(isx_non_str) == 1:
            hp = args[isx_non_str[0]]
            cand_names = [k for k, v in init_args.items() if v == hp]
            obj._hparams_name = cand_names[0] if cand_names else None
        else:
            hp = {arg: init_args[arg] for arg in args if isinstance(arg, str)}
            obj._hparams_name = "kwargs"

    # `hparams` are expected here
    if hp:
        obj._set_hparams(hp)
    # make deep copy so  there is not other runtime changes reflected
    obj._hparams_initial = copy.deepcopy(obj._hparams)


class AttributeDict(Dict):
    """Extended dictionary accesisable with dot notation.

    >>> ad = AttributeDict({'key1': 1, 'key2': 'abc'})
    >>> ad.key1
    1
    >>> ad.update({'my-key': 3.14})
    >>> ad.update(mew_key=42)
    >>> ad.key1 = 2
    >>> ad
    "key1":    2
    "key2":    abc
    "mew_key": 42
    "my-key":  3.14
    """

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exp:
            raise AttributeError(f'Missing attribute "{key}"') from exp

    def __setattr__(self, key, val):
        self[key] = val

    def __repr__(self):
        if not len(self):
            return ""
        max_key_length = max([len(str(k)) for k in self])
        tmp_name = '{:' + str(max_key_length + 3) + 's} {}'
        rows = [tmp_name.format(f'"{n}":', self[n]) for n in sorted(self.keys())]
        out = '\n'.join(rows)
        return out


def _lightning_get_all_attr_holders(model, attribute):
    """
    Special attribute finding for Lightning. Gets all of the objects or dicts that holds attribute.
    Checks for attribute in model namespace, the old hparams namespace/dict, and the datamodule.
    """
    trainer = getattr(model, 'trainer', None)

    holders = []

    # Check if attribute in model
    if hasattr(model, attribute):
        holders.append(model)

    # Check if attribute in model.hparams, either namespace or dict
    if hasattr(model, 'hparams'):
        if attribute in model.hparams:
            holders.append(model.hparams)

    # Check if the attribute in datamodule (datamodule gets registered in Trainer)
    if trainer is not None and trainer.datamodule is not None and hasattr(trainer.datamodule, attribute):
        holders.append(trainer.datamodule)

    return holders


def _lightning_get_first_attr_holder(model, attribute):
    """
    Special attribute finding for Lightning.  Gets the object or dict that holds attribute, or None.
    Checks for attribute in model namespace, the old hparams namespace/dict, and the datamodule,
    returns the last one that has it.
    """
    holders = _lightning_get_all_attr_holders(model, attribute)
    if len(holders) == 0:
        return None
    # using the last holder to preserve backwards compatibility
    return holders[-1]


def lightning_hasattr(model, attribute):
    """
    Special hasattr for Lightning. Checks for attribute in model namespace,
    the old hparams namespace/dict, and the datamodule.
    """
    return _lightning_get_first_attr_holder(model, attribute) is not None


def lightning_getattr(model, attribute):
    """
    Special getattr for Lightning. Checks for attribute in model namespace,
    the old hparams namespace/dict, and the datamodule.

    Raises:
        AttributeError:
            If ``model`` doesn't have ``attribute`` in any of
            model namespace, the hparams namespace/dict, and the datamodule.
    """
    holder = _lightning_get_first_attr_holder(model, attribute)
    if holder is None:
        raise AttributeError(
            f'{attribute} is neither stored in the model namespace'
            ' nor the `hparams` namespace/dict, nor the datamodule.'
        )

    if isinstance(holder, dict):
        return holder[attribute]
    return getattr(holder, attribute)


def lightning_setattr(model, attribute, value):
    """
    Special setattr for Lightning. Checks for attribute in model namespace
    and the old hparams namespace/dict.
    Will also set the attribute on datamodule, if it exists.

    Raises:
        AttributeError:
            If ``model`` doesn't have ``attribute`` in any of
            model namespace, the hparams namespace/dict, and the datamodule.
    """
    holders = _lightning_get_all_attr_holders(model, attribute)
    if len(holders) == 0:
        raise AttributeError(
            f'{attribute} is neither stored in the model namespace'
            ' nor the `hparams` namespace/dict, nor the datamodule.'
        )

    for holder in holders:
        if isinstance(holder, dict):
            holder[attribute] = value
        else:
            setattr(holder, attribute, value)
