import ast
import csv
import inspect
import os

import torch
import yaml
from argparse import Namespace
from typing import Union, Dict, Any, Optional, Callable

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import rank_zero_warn, AttributeDict
from pytorch_lightning.utilities.io import load as pl_load

PRIMITIVE_TYPES = (bool, int, float, str)
ALLOWED_CONFIG_TYPES = (AttributeDict, dict, Namespace)
try:
    from omegaconf import Container
except ImportError:
    pass
else:
    ALLOWED_CONFIG_TYPES = ALLOWED_CONFIG_TYPES + (Container, )


class ModelIO(object):
    CHECKPOINT_KEY_HYPER_PARAMS = 'hyper_parameters'
    CHECKPOINT_NAME_HYPER_PARAMS = 'hparams_name'

    @classmethod
    def load_from_metrics(cls, weights_path, tags_csv, map_location=None):
        r"""
        Warning:
            Deprecated in version 0.7.0. You should use :meth:`load_from_checkpoint` instead.
            Will be removed in v0.9.0.
        """
        rank_zero_warn(
            "`load_from_metrics` method has been unified with `load_from_checkpoint` in v0.7.0."
            " The deprecated method will be removed in v0.9.0.", DeprecationWarning
        )
        return cls.load_from_checkpoint(weights_path, tags_csv=tags_csv, map_location=map_location)

    @classmethod
    def load_from_checkpoint(
            cls,
            checkpoint_path: str,
            *args,
            map_location: Optional[Union[Dict[str, str], str, torch.device, int, Callable]] = None,
            hparams_file: Optional[str] = None,
            tags_csv: Optional[str] = None,  # backward compatible, todo: remove in v0.9.0
            **kwargs
    ):
        r"""
        Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint
        it stores the arguments passed to `__init__`  in the checkpoint under `module_arguments`

        Any arguments specified through \*args and \*\*kwargs will override args stored in `hparams`.

        Args:
            checkpoint_path: Path to checkpoint. This can also be a URL.
            args: Any positional args needed to init the model.
            map_location:
                If your checkpoint saved a GPU model and you now load on CPUs
                or a different number of GPUs, use this to map to the new setup.
                The behaviour is the same as in :func:`torch.load`.
            hparams_file: Optional path to a .yaml file with hierarchical structure
                as in this example::

                    drop_prob: 0.2
                    dataloader:
                        batch_size: 32

                You most likely won't need this since Lightning will always save the hyperparameters
                to the checkpoint.
                However, if your checkpoint weights don't have the hyperparameters saved,
                use this method to pass in a .yaml file with the hparams you'd like to use.
                These will be converted into a :class:`~dict` and passed into your
                :class:`LightningModule` for use.

                If your model's `hparams` argument is :class:`~argparse.Namespace`
                and .yaml file has hierarchical structure, you need to refactor your model to treat
                `hparams` as :class:`~dict`.

                .csv files are acceptable here till v0.9.0, see tags_csv argument for detailed usage.
            tags_csv:
                .. warning:: .. deprecated:: 0.7.6

                    `tags_csv` argument is deprecated in v0.7.6. Will be removed v0.9.0.

                Optional path to a .csv file with two columns (key, value)
                as in this example::

                    key,value
                    drop_prob,0.2
                    batch_size,32

                Use this method to pass in a .csv file with the hparams you'd like to use.
            hparam_overrides: A dictionary with keys to override in the hparams
            kwargs: Any keyword args needed to init the model.

        Return:
            :class:`LightningModule` with loaded weights and hyperparameters (if available).

        Example:
            .. code-block:: python

                # load weights without mapping ...
                MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

                # or load weights mapping all weights from GPU 1 to GPU 0 ...
                map_location = {'cuda:1':'cuda:0'}
                MyLightningModule.load_from_checkpoint(
                    'path/to/checkpoint.ckpt',
                    map_location=map_location
                )

                # or load weights and hyperparameters from separate files.
                MyLightningModule.load_from_checkpoint(
                    'path/to/checkpoint.ckpt',
                    hparams_file='/path/to/hparams_file.yaml'
                )

                # override some of the params with new values
                MyLightningModule.load_from_checkpoint(
                    PATH,
                    num_layers=128,
                    pretrained_ckpt_path: NEW_PATH,
                )

                # predict
                pretrained_model.eval()
                pretrained_model.freeze()
                y_hat = pretrained_model(x)
        """
        if map_location is not None:
            checkpoint = pl_load(checkpoint_path, map_location=map_location)
        else:
            checkpoint = pl_load(checkpoint_path, map_location=lambda storage, loc: storage)

        # add the hparams from csv file to checkpoint
        if tags_csv is not None:
            hparams_file = tags_csv
            rank_zero_warn('`tags_csv` argument is deprecated in v0.7.6. Will be removed v0.9.0', DeprecationWarning)

        if hparams_file is not None:
            extension = hparams_file.split('.')[-1]
            if extension.lower() in ('csv'):
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ('yml', 'yaml'):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError('.csv, .yml or .yaml is required for `hparams_file`')

            hparams['on_gpu'] = False

            # overwrite hparams by the given file
            checkpoint[cls.CHECKPOINT_KEY_HYPER_PARAMS] = hparams

        # override the module_arguments with values that were passed in
        checkpoint[cls.CHECKPOINT_KEY_HYPER_PARAMS].update(kwargs)

        model = cls._load_model_state(checkpoint, *args, **kwargs)
        return model

    @classmethod
    def _load_model_state(cls, checkpoint: Dict[str, Any], *args, **kwargs):
        # pass in the values we saved automatically
        if cls.CHECKPOINT_KEY_HYPER_PARAMS in checkpoint:
            # todo add some back compatibility
            model_args = checkpoint[cls.CHECKPOINT_KEY_HYPER_PARAMS]
            args_name = checkpoint.get(cls.CHECKPOINT_NAME_HYPER_PARAMS)
            init_args_name = inspect.signature(cls).parameters.keys()
            if args_name == 'kwargs':
                cls_kwargs = {k: v for k, v in model_args.items() if k in init_args_name}
                kwargs.update(**cls_kwargs)
            elif args_name:
                if args_name in init_args_name:
                    kwargs.update({args_name: model_args})
            else:
                args = (model_args, ) + args

        # load the state_dict on the model automatically
        model = cls(*args, **kwargs)
        model.load_state_dict(checkpoint['state_dict'])

        # give model a chance to load something
        model.on_load_checkpoint(checkpoint)

        return model

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Do something with the checkpoint.
        Gives model a chance to load something before ``state_dict`` is restored.

        Args:
            checkpoint: A dictionary with variables from the checkpoint.
        """

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Give the model a chance to add something to the checkpoint.
        ``state_dict`` is already there.

        Args:
            checkpoint: A dictionary in which you can save variables to save in a checkpoint.
                Contents need to be pickleable.
        """

    # -------------------------
    # OPTIONAL HOOKS
    # -------------------------
    def on_hpc_save(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook to do whatever you need right before Slurm manager saves the model.

        Args:
            checkpoint: A dictionary in which you can save variables to save in a checkpoint.
                Contents need to be pickleable.
        """

    def on_hpc_load(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook to do whatever you need right before Slurm manager loads the model.

        Args:
            checkpoint: A dictionary with variables from the checkpoint.
        """


def update_hparams(hparams: dict, updates: dict) -> None:
    """
    Overrides hparams with new values

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


def load_hparams_from_tags_csv(tags_csv: str) -> Dict[str, Any]:
    """Load hparams from a file.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_csv = './testing-hparams.csv'
    >>> save_hparams_to_tags_csv(path_csv, hparams)
    >>> hparams_new = load_hparams_from_tags_csv(path_csv)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_csv)
    """
    if not os.path.isfile(tags_csv):
        rank_zero_warn(f'Missing Tags: {tags_csv}.', RuntimeWarning)
        return {}

    with open(tags_csv) as fp:
        csv_reader = csv.reader(fp, delimiter=',')
        tags = {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}

    return tags


def save_hparams_to_tags_csv(tags_csv: str, hparams: Union[dict, Namespace]) -> None:
    if not os.path.isdir(os.path.dirname(tags_csv)):
        raise RuntimeError(f'Missing folder: {os.path.dirname(tags_csv)}.')

    if isinstance(hparams, Namespace):
        hparams = vars(hparams)

    with open(tags_csv, 'w') as fp:
        fieldnames = ['key', 'value']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writerow({'key': 'key', 'value': 'value'})
        for k, v in hparams.items():
            writer.writerow({'key': k, 'value': v})


def load_hparams_from_yaml(config_yaml: str) -> Dict[str, Any]:
    """Load hparams from a file.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_yaml = './testing-hparams.yaml'
    >>> save_hparams_to_yaml(path_yaml, hparams)
    >>> hparams_new = load_hparams_from_yaml(path_yaml)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_yaml)
    """
    if not os.path.isfile(config_yaml):
        rank_zero_warn(f'Missing Tags: {config_yaml}.', RuntimeWarning)
        return {}

    with open(config_yaml) as fp:
        tags = yaml.load(fp, Loader=yaml.SafeLoader)

    return tags


def save_hparams_to_yaml(config_yaml, hparams: Union[dict, Namespace]) -> None:
    if not os.path.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f'Missing folder: {os.path.dirname(config_yaml)}.')

    if isinstance(hparams, Namespace):
        hparams = vars(hparams)

    with open(config_yaml, 'w', newline='') as fp:
        yaml.dump(hparams, fp)


def convert(val: str) -> Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as err:
        log.debug(err)
        return val
