from pytorch_lightning.utilities.distributed import rank_zero_warn


class WarningCache:

    def __init__(self):
        self.warnings = set()

    def warn(self, m):
        if m not in self.warnings:
            self.warnings.add(m)
            rank_zero_warn(m)
