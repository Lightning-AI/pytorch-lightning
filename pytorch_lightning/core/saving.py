import ast
import csv
import os
import yaml
from argparse import Namespace
from typing import Union, Dict, Any

from pytorch_lightning import _logger as log
from pytorch_lightning.utilities import rank_zero_warn


class ModelIO(object):

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
    except (ValueError, SyntaxError) as e:
        log.debug(e)
        return val
