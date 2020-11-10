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
import pickle
from argparse import Namespace
from typing import Dict, Union

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
    except (pickle.PicklingError, AttributeError):
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


def get_init_args(frame) -> dict:
    _, _, _, local_vars = inspect.getargvalues(frame)
    if '__class__' not in local_vars:
        return
    cls = local_vars['__class__']
    spec = inspect.getfullargspec(cls.__init__)
    init_parameters = inspect.signature(cls.__init__).parameters
    self_identifier = spec.args[0]  # "self" unless user renames it (always first arg)
    varargs_identifier = spec.varargs  # by convention this is named "*args"
    kwargs_identifier = spec.varkw  # by convention this is named "**kwargs"
    exclude_argnames = (
        varargs_identifier, kwargs_identifier, self_identifier, '__class__', 'frame', 'frame_args'
    )

    # only collect variables that appear in the signature
    local_args = {k: local_vars[k] for k in init_parameters.keys()}
    local_args.update(local_args.get(kwargs_identifier, {}))
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


def lightning_hasattr(model, attribute):
    """ Special hasattr for lightning. Checks for attribute in model namespace,
        the old hparams namespace/dict, and the datamodule. """
    trainer = getattr(model, 'trainer', None)

    attr = False
    # Check if attribute in model
    if hasattr(model, attribute):
        attr = True
    # Check if attribute in model.hparams, either namespace or dict
    elif hasattr(model, 'hparams'):
        if isinstance(model.hparams, dict):
            attr = attribute in model.hparams
        else:
            attr = hasattr(model.hparams, attribute)
    # Check if the attribute in datamodule (datamodule gets registered in Trainer)
    if not attr and trainer is not None:
        attr = hasattr(trainer.datamodule, attribute)

    return attr


def lightning_getattr(model, attribute):
    """ Special getattr for lightning. Checks for attribute in model namespace,
        the old hparams namespace/dict, and the datamodule. """
    trainer = getattr(model, 'trainer', None)

    # Check if attribute in model
    if hasattr(model, attribute):
        attr = getattr(model, attribute)
    # Check if attribute in model.hparams, either namespace or dict
    elif hasattr(model, 'hparams') and isinstance(model.hparams, dict) and attribute in model.hparams:
        attr = model.hparams[attribute]
    elif hasattr(model, 'hparams') and hasattr(model.hparams, attribute):
        attr = getattr(model.hparams, attribute)
    # Check if the attribute in datamodule (datamodule gets registered in Trainer)
    elif trainer is not None and trainer.datamodule is not None and hasattr(trainer.datamodule, attribute):
        attr = getattr(trainer.datamodule, attribute)
    else:
        raise ValueError(f'{attribute} is neither stored in the model namespace'
                         ' nor the `hparams` namespace/dict, nor the datamodule.')
    return attr


def lightning_setattr(model, attribute, value):
    """ Special setattr for lightning. Checks for attribute in model namespace
        and the old hparams namespace/dict.
        Will also set the attribute on datamodule, if it exists.
    """
    if not lightning_hasattr(model, attribute):
        raise ValueError(f'{attribute} is neither stored in the model namespace'
                         ' nor the `hparams` namespace/dict, nor the datamodule.')

    trainer = getattr(model, 'trainer', None)

    # Check if attribute in model
    if hasattr(model, attribute):
        setattr(model, attribute, value)

    # Check if attribute in model.hparams, either namespace or dict
    elif hasattr(model, 'hparams'):
        if isinstance(model.hparams, dict):
            model.hparams[attribute] = value
        else:
            setattr(model.hparams, attribute, value)

    # Check if the attribute in datamodule (datamodule gets registered in Trainer)
    if trainer is not None and trainer.datamodule is not None and hasattr(trainer.datamodule, attribute):
        setattr(trainer.datamodule, attribute, value)
