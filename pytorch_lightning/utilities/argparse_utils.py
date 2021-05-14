from pytorch_lightning.utilities import rank_zero_deprecation

rank_zero_deprecation("`argparse_utils` package has been renamed to `argparse` since v1.2 and will be removed in v1.4")

# for backward compatibility with old checkpoints (versions < 1.2.0)
# that need to be able to unpickle the function from the checkpoint
from pytorch_lightning.utilities.argparse import _gpus_arg_default  # noqa: E402 F401 # isort: skip
