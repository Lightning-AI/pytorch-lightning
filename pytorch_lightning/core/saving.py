import ast
import collections
import csv
import os
import yaml
from argparse import Namespace
from typing import Union, Dict, Any, List

from pytorch_lightning import _logger as log


class ModelIO(object):

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Do something with the checkpoint
        Gives model a chance to load something before state_dict is restored
        :param checkpoint:
        :return:
        """

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        Give the model a chance to add something to the checkpoint.
        state_dict is already there
        """

    # -------------------------
    # OPTIONAL HOOKS
    # -------------------------
    def on_hpc_save(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook to do whatever you need right before Slurm manager saves the model
        """

    def on_hpc_load(self, checkpoint: Dict[str, Any]) -> None:
        """
        Hook to do whatever you need right before Slurm manager loads the model
        """


def load_hparams_from_tags_csv(tags_csv: str) -> Dict[str, Any]:
    if not os.path.isfile(tags_csv):
        log.warning(f'Missing Tags: {tags_csv}.')
        return {}

    with open(tags_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        tags = {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}

    return tags


def load_hparams_from_yaml(config_yaml: str) -> Dict[str, Any]:
    if not os.path.isfile(config_yaml):
        log.warning(f'Missing Tags: {config_yaml}.')
        return {}

    with open(config_yaml) as f:
       tags = yaml.load(f, Loader=yaml.SafeLoader)

    return tags


def convert(val: str) -> Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except ValueError:
        return val
