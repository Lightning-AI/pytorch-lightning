import os
import csv
import logging as log
from argparse import Namespace


class ModelIO(object):

    def on_load_checkpoint(self, checkpoint):
        """
        Do something with the checkpoint
        Gives model a chance to load something before state_dict is restored
        :param checkpoint:
        :return:
        """

    def on_save_checkpoint(self, checkpoint):
        """
        Give the model a chance to add something to the checkpoint.
        state_dict is already there
        """

    # -------------------------
    # OPTIONAL HOOKS
    # -------------------------
    def on_hpc_save(self, checkpoint):
        """
        Hook to do whatever you need right before Slurm manager saves the model
        :return:
        """

    def on_hpc_load(self, checkpoint):
        """
        Hook to do whatever you need right before Slurm manager loads the model
        :return:
        """


def load_hparams_from_tags_csv(tags_csv) -> Namespace:
    if not os.path.isfile(tags_csv):
        log.warning(f'Missing Tags: {tags_csv}.')
        return Namespace()

    with open(tags_csv) as f:
        csv_reader = csv.reader(f, delimiter=',')
        tags = {row[0]: convert(row[1]) for row in list(csv_reader)[1:]}
    ns = Namespace(**tags)
    return ns


def convert(val):
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
