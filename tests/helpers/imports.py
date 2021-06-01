import operator

from pytorch_lightning.utilities.imports import _compare_version

if _compare_version("torchtext", operator.ge, "0.9.0"):
    from torchtext.legacy.data import Batch, Dataset, Example, Field, Iterator, LabelField  # noqa: F401
else:
    from torchtext.data import Batch, Dataset, Example, Field, Iterator, LabelField  # noqa: F401
