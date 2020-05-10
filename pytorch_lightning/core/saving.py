import ast
import csv
import os
import yaml
import logging
from argparse import Namespace
from typing import Union, Dict, Any

from pytorch_lightning import _logger as log


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
    if not os.path.isfile(tags_csv):
        log.warning(f'Missing Tags: {tags_csv}.')
        return {}

    with open(tags_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        tags = {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}

    return tags


def save_hparams_to_tags_csv(tags_csv: str, hparams: dict) -> None:
    if not os.path.isdir(os.path.dirname(tags_csv)):
        raise RuntimeError(f'Missing folder: {os.path.dirname(tags_csv)}.')
    with open(tags_csv, 'w') as f:
        fieldnames = ['key', 'value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({'key': 'key', 'value': 'value'})
        for k, v in hparams.items():
            writer.writerow({'key': k, 'value': v})


def load_hparams_from_yaml(config_yaml: str) -> Dict[str, Any]:
    if not os.path.isfile(config_yaml):
        rank_zero_warn(f'Missing Tags: {config_yaml}.', RuntimeWarning)
        return {}
    with open(config_yaml) as f:
        tags = yaml.load(f, Loader=yaml.SafeLoader)
    return tags


def save_hparams_to_yaml(config_yaml, hparams: dict) -> None:
    if not os.path.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f'Missing folder: {os.path.dirname(config_yaml)}.')
    with open(config_yaml, 'w', newline='') as f:
        yaml.dump(hparams, f)


def convert(val: str) -> Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError) as e:
        logging.debug(e)
        return val
