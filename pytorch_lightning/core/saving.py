import csv
import os
from argparse import Namespace
from typing import Union, Dict, Any

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


def load_hparams_from_tags_csv(tags_csv: str) -> Namespace:
    if not os.path.isfile(tags_csv):
        log.warning(f'Missing Tags: {tags_csv}.')
        return Namespace()

    with open(tags_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        tags = {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}
    ns = Namespace(**tags)
    return ns


def convert(val: str) -> Union[int, float, bool, str]:
    constructors = [int, float, str]

    if isinstance(val, str):
        if val.lower() == 'true':
            return True
        if val.lower() == 'false':
            return False

    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass
    return val
