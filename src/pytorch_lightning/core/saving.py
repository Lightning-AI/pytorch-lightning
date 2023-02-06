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

import ast
import csv
import inspect
import logging
import os
from argparse import Namespace
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Any, Callable, cast, Dict, IO, MutableMapping, Optional, Type, Union
from warnings import warn

import yaml
from lightning_utilities.core.apply_func import apply_to_collection
from typing_extensions import Self

import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import _load as pl_load
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _MAP_LOCATION_TYPE, _PATH
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.migration.utils import _pl_migrate_checkpoint
from pytorch_lightning.utilities.parsing import AttributeDict, parse_class_init_keys
from pytorch_lightning.utilities.rank_zero import rank_zero_warn

log = logging.getLogger(__name__)
PRIMITIVE_TYPES = (bool, int, float, str)
ALLOWED_CONFIG_TYPES = (AttributeDict, MutableMapping, Namespace)

if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf
    from omegaconf.dictconfig import DictConfig
    from omegaconf.errors import UnsupportedValueType, ValidationError

# the older shall be on the top
CHECKPOINT_PAST_HPARAMS_KEYS = ("hparams", "module_arguments")  # used in 0.7.6


class ModelIO:
    CHECKPOINT_HYPER_PARAMS_KEY = "hyper_parameters"
    CHECKPOINT_HYPER_PARAMS_NAME = "hparams_name"
    CHECKPOINT_HYPER_PARAMS_TYPE = "hparams_type"

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[_PATH, IO],
        map_location: _MAP_LOCATION_TYPE = None,
        hparams_file: Optional[_PATH] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> Self:  # type: ignore[valid-type]
        r"""
        Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint
        it stores the arguments passed to ``__init__``  in the checkpoint under ``"hyper_parameters"``.

        Any arguments specified through \*\*kwargs will override args stored in ``"hyper_parameters"``.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.
            hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
                as in this example::

                    drop_prob: 0.2
                    dataloader:
                        batch_size: 32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                These will be converted into a :class:`~dict` and passed into your
                :class:`LightningModule` for use.

                If your model's ``hparams`` argument is :class:`~argparse.Namespace`
                and ``.yaml`` file has hierarchical structure, you need to refactor your model to treat
                ``hparams`` as :class:`~dict`.
            strict: Whether to strictly enforce that the keys in :attr:`checkpoint_path` match the keys
                returned by this module's state dict.
            \**kwargs: Any extra keyword args needed to init the model. Can also be used to override saved
                hyperparameter values.

        Return:
            :class:`LightningModule` instance with loaded weights and hyperparameters (if available).

        Note:
            ``load_from_checkpoint`` is a **class** method. You should use your :class:`LightningModule`
            **class** to call it instead of the :class:`LightningModule` instance.

        Example::

            # load weights without mapping ...
            model = MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

            # or load weights mapping all weights from GPU 1 to GPU 0 ...
            map_location = {'cuda:1':'cuda:0'}
            model = MyLightningModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                map_location=map_location
            )

            # or load weights and hyperparameters from separate files.
            model = MyLightningModule.load_from_checkpoint(
                'path/to/checkpoint.ckpt',
                hparams_file='/path/to/hparams_file.yaml'
            )

            # override some of the params with new values
            model = MyLightningModule.load_from_checkpoint(
                PATH,
                num_layers=128,
                pretrained_ckpt_path=NEW_PATH,
            )

            # predict
            pretrained_model.eval()
            pretrained_model.freeze()
            y_hat = pretrained_model(x)
        """
        return _load_from_checkpoint(
            cls,
            checkpoint_path,
            map_location,
            hparams_file,
            strict,
            **kwargs,
        )


def _load_from_checkpoint(
    cls: Union[Type["ModelIO"], Type["pl.LightningModule"], Type["pl.LightningDataModule"]],
    checkpoint_path: Union[_PATH, IO],
    map_location: _MAP_LOCATION_TYPE = None,
    hparams_file: Optional[_PATH] = None,
    strict: Optional[bool] = None,
    **kwargs: Any,
) -> Union["pl.LightningModule", "pl.LightningDataModule"]:
    if map_location is None:
        map_location = cast(_MAP_LOCATION_TYPE, lambda storage, loc: storage)
    with pl_legacy_patch():
        checkpoint = pl_load(checkpoint_path, map_location=map_location)

    # convert legacy checkpoints to the new format
    checkpoint = _pl_migrate_checkpoint(
        checkpoint, checkpoint_path=(checkpoint_path if isinstance(checkpoint_path, (str, Path)) else None)
    )

    if hparams_file is not None:
        extension = str(hparams_file).split(".")[-1]
        if extension.lower() == "csv":
            hparams = load_hparams_from_tags_csv(hparams_file)
        elif extension.lower() in ("yml", "yaml"):
            hparams = load_hparams_from_yaml(hparams_file)
        else:
            raise ValueError(".csv, .yml or .yaml is required for `hparams_file`")

        # overwrite hparams by the given file
        checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY] = hparams

    # TODO: make this a migration:
    # for past checkpoint need to add the new key
    checkpoint.setdefault(cls.CHECKPOINT_HYPER_PARAMS_KEY, {})
    # override the hparams with values that were passed in
    checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)

    if issubclass(cls, pl.LightningDataModule):
        return _load_state(cls, checkpoint, **kwargs)
    if issubclass(cls, pl.LightningModule):
        return _load_state(cls, checkpoint, strict=strict, **kwargs)
    raise NotImplementedError(f"Unsupported {cls}")


def _load_state(
    cls: Union[Type["pl.LightningModule"], Type["pl.LightningDataModule"]],
    checkpoint: Dict[str, Any],
    strict: Optional[bool] = None,
    **cls_kwargs_new: Any,
) -> Union["pl.LightningModule", "pl.LightningDataModule"]:
    cls_spec = inspect.getfullargspec(cls.__init__)
    cls_init_args_name = inspect.signature(cls.__init__).parameters.keys()

    self_var, args_var, kwargs_var = parse_class_init_keys(cls)
    drop_names = [n for n in (self_var, args_var, kwargs_var) if n]
    cls_init_args_name = list(filter(lambda n: n not in drop_names, cls_init_args_name))

    cls_kwargs_loaded = {}
    # pass in the values we saved automatically
    if cls.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:

        if issubclass(cls, pl.LightningModule):
            # TODO: make this a migration:
            # 1. (backward compatibility) Try to restore model hparams from checkpoint using old/past keys
            for _old_hparam_key in CHECKPOINT_PAST_HPARAMS_KEYS:
                cls_kwargs_loaded.update(checkpoint.get(_old_hparam_key, {}))

        # 2. Try to restore model hparams from checkpoint using the new key
        cls_kwargs_loaded.update(checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_KEY, {}))

        # 3. Ensure that `cls_kwargs_old` has the right type, back compatibility between dict and Namespace
        cls_kwargs_loaded = _convert_loaded_hparams(cls_kwargs_loaded, checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_TYPE))

        # 4. Update cls_kwargs_new with cls_kwargs_old, such that new has higher priority
        args_name = checkpoint.get(cls.CHECKPOINT_HYPER_PARAMS_NAME)
        if args_name and args_name in cls_init_args_name:
            cls_kwargs_loaded = {args_name: cls_kwargs_loaded}

    _cls_kwargs = {}
    _cls_kwargs.update(cls_kwargs_loaded)
    _cls_kwargs.update(cls_kwargs_new)

    if not cls_spec.varkw:
        # filter kwargs according to class init unless it allows any argument via kwargs
        _cls_kwargs = {k: v for k, v in _cls_kwargs.items() if k in cls_init_args_name}

    obj = cls(**_cls_kwargs)

    if isinstance(obj, pl.LightningModule):
        # give model a chance to load something
        obj.on_load_checkpoint(checkpoint)

    if isinstance(obj, pl.LightningDataModule):
        if obj.__class__.__qualname__ in checkpoint:
            obj.load_state_dict(checkpoint[obj.__class__.__qualname__])
        return obj

    # load the state_dict on the model automatically
    assert strict is not None
    keys = obj.load_state_dict(checkpoint["state_dict"], strict=strict)

    if not strict:
        if keys.missing_keys:
            rank_zero_warn(
                f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
            )
        if keys.unexpected_keys:
            rank_zero_warn(
                f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
            )

    return obj


def _convert_loaded_hparams(
    model_args: Dict[str, Any], hparams_type: Optional[Union[Callable, str]] = None
) -> Dict[str, Any]:
    """Convert hparams according given type in callable or string (past) format."""
    # if not hparams type define
    if not hparams_type:
        return model_args
    # if past checkpoint loaded, convert str to callable
    if isinstance(hparams_type, str):
        hparams_type = AttributeDict
    # convert hparams
    return hparams_type(model_args)


def update_hparams(hparams: dict, updates: dict) -> None:
    """Overrides hparams with new values.

    >>> hparams = {'c': 4}
    >>> update_hparams(hparams, {'a': {'b': 2}, 'c': 1})
    >>> hparams['a']['b'], hparams['c']
    (2, 1)
    >>> update_hparams(hparams, {'a': {'b': 4}, 'c': 7})
    >>> hparams['a']['b'], hparams['c']
    (4, 7)

    Args:
        hparams: the original params and also target object
        updates: new params to be used as update
    """
    for k, v in updates.items():
        # if missing, add the key
        if k not in hparams:
            hparams[k] = v
            continue

        # recurse if dictionary
        if isinstance(v, dict):
            update_hparams(hparams[k], updates[k])
        else:
            # update the value
            hparams.update({k: v})


def load_hparams_from_tags_csv(tags_csv: _PATH) -> Dict[str, Any]:
    """Load hparams from a file.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_csv = os.path.join('.', 'testing-hparams.csv')
    >>> save_hparams_to_tags_csv(path_csv, hparams)
    >>> hparams_new = load_hparams_from_tags_csv(path_csv)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_csv)
    """
    fs = get_filesystem(tags_csv)
    if not fs.exists(tags_csv):
        rank_zero_warn(f"Missing Tags: {tags_csv}.", category=RuntimeWarning)
        return {}

    with fs.open(tags_csv, "r", newline="") as fp:
        csv_reader = csv.reader(fp, delimiter=",")
        tags = {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}

    return tags


def save_hparams_to_tags_csv(tags_csv: _PATH, hparams: Union[dict, Namespace]) -> None:
    fs = get_filesystem(tags_csv)
    if not fs.isdir(os.path.dirname(tags_csv)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(tags_csv)}.")

    if isinstance(hparams, Namespace):
        hparams = vars(hparams)

    with fs.open(tags_csv, "w", newline="") as fp:
        fieldnames = ["key", "value"]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writerow({"key": "key", "value": "value"})
        for k, v in hparams.items():
            writer.writerow({"key": k, "value": v})


def load_hparams_from_yaml(config_yaml: _PATH, use_omegaconf: bool = True) -> Dict[str, Any]:
    """Load hparams from a file.

        Args:
            config_yaml: Path to config yaml file
            use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
                the hparams will be converted to ``DictConfig`` if possible.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_yaml = './testing-hparams.yaml'
    >>> save_hparams_to_yaml(path_yaml, hparams)
    >>> hparams_new = load_hparams_from_yaml(path_yaml)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_yaml)
    """
    fs = get_filesystem(config_yaml)
    if not fs.exists(config_yaml):
        rank_zero_warn(f"Missing Tags: {config_yaml}.", category=RuntimeWarning)
        return {}

    with fs.open(config_yaml, "r") as fp:
        hparams = yaml.full_load(fp)

    if _OMEGACONF_AVAILABLE:
        if use_omegaconf:
            try:
                return OmegaConf.create(hparams)
            except (UnsupportedValueType, ValidationError):
                pass
    return hparams


def save_hparams_to_yaml(config_yaml: _PATH, hparams: Union[dict, Namespace], use_omegaconf: bool = True) -> None:
    """
    Args:
        config_yaml: path to new YAML file
        hparams: parameters to be saved
        use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
            the hparams will be converted to ``DictConfig`` if possible.

    """
    fs = get_filesystem(config_yaml)
    if not fs.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")

    # convert Namespace or AD to dict
    if isinstance(hparams, Namespace):
        hparams = vars(hparams)
    elif isinstance(hparams, AttributeDict):
        hparams = dict(hparams)

    # saving with OmegaConf objects
    if _OMEGACONF_AVAILABLE and use_omegaconf:
        # deepcopy: hparams from user shouldn't be resolved
        hparams = deepcopy(hparams)
        hparams = apply_to_collection(hparams, DictConfig, OmegaConf.to_container, resolve=True)
        with fs.open(config_yaml, "w", encoding="utf-8") as fp:
            try:
                OmegaConf.save(hparams, fp)
                return
            except (UnsupportedValueType, ValidationError):
                pass

    if not isinstance(hparams, dict):
        raise TypeError("hparams must be dictionary")

    hparams_allowed = {}
    # drop parameters which contain some strange datatypes as fsspec
    for k, v in hparams.items():
        try:
            v = v.name if isinstance(v, Enum) else v
            yaml.dump(v)
        except TypeError:
            warn(f"Skipping '{k}' parameter because it is not possible to safely dump to YAML.")
            hparams[k] = type(v).__name__
        else:
            hparams_allowed[k] = v

    # saving the standard way
    with fs.open(config_yaml, "w", newline="") as fp:
        yaml.dump(hparams_allowed, fp)


def convert(val: str) -> Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as err:
        log.debug(err)
        return val
