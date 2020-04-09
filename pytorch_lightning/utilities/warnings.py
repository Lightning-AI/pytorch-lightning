"""Cutome Ligthning warnings"""

import warnings


def rank_zero_warn(*args, trainer=None, **kwargs):
    if trainer and (trainer.proc_rank() > 0 or trainer.node_rank > 0):
        return
    warnings.warn(*args, **kwargs)
