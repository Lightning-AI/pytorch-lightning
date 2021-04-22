from distutils.version import LooseVersion

import pytorch_lightning.utilities.argparse


def get_version(checkpoint: dict) -> str:
    return checkpoint["pytorch-lightning_version"]


def set_version(checkpoint: dict, version: str):
    checkpoint["pytorch-lightning_version"] = version


def should_upgrade(checkpoint: dict, target: str) -> bool:
    return LooseVersion(get_version(checkpoint)) < LooseVersion(target)


class pl_legacy_patch:
    """
    Registers legacy artifacts (classes, methods, etc.) that were removed but still need to be
    included for unpickling old checkpoints.
    """

    def __enter__(self):
        setattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default", lambda x: x)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        delattr(pytorch_lightning.utilities.argparse, "_gpus_arg_default")
