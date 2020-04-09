"""Cutome Ligthning warnings"""

import warnings

from pytorch_lightning.loggers import rank_zero_only


class RankWarning:
    """Warning depending on ranks,"""

    def __init__(self):
        self._rank = 0

    @property
    def rank(self) -> int:
        """Process rank. In general, metrics should only be logged by the process with rank 0."""
        return self._rank

    @rank.setter
    def rank(self, value: int) -> None:
        """Set the process rank."""
        self._rank = value

    @rank_zero_only
    def __call__(self, *args, **kwargs):
        warnings.warn(*args, **kwargs)
