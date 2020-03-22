import ast
import collections
import csv
import logging as log
import os
from argparse import Namespace
from typing import Union, Dict, Any, List


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
    tags = {}

    if not os.path.isfile(tags_csv):
        log.warning(f'Missing Tags: {tags_csv}.')
        return tags

    with open(tags_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        for key, value in list(csv_reader)[1:]:
            value = convert(value)
            merge_dict(tags, hierarchize(key.split('/'), value))

    return tags


def convert(val: str) -> Union[int, float, bool, str]:
    try:
        return ast.literal_eval(val)
    except ValueError:
        return val


def hierarchize(keys: List[str], value: Optional[int, float, bool, str] = None) -> Dict[str, Any]:
    if len(keys) == 1:
        return {keys[0]: value}
    else:
        return {keys[0]: hierarchize(keys[1:], value)}


def merge_dict(base_dict: Dict[str: Any], input_dict: Dict[str: Any]) -> None:
    for k, v in input_dict.items():
        if k in base_dict and isinstance(base_dict[k], dict) and isinstance(input_dict[k], dict):
            merge_dict(base_dict[k], input_dict[k])
        else:
            base_dict[k] = input_dict[k]