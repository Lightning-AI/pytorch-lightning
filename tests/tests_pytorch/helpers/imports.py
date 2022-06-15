import operator

from pytorch_lightning.utilities.imports import _compare_version, _TORCHTEXT_LEGACY

if _TORCHTEXT_LEGACY:
    if _compare_version("torchtext", operator.ge, "0.9.0"):
        from torchtext.legacy.data import Batch, Dataset, Example, Field, Iterator, LabelField
    else:
        from torchtext.data import Batch, Dataset, Example, Field, Iterator, LabelField
else:
    Batch = type(None)
    Dataset = type(None)
    Example = type(None)
    Field = type(None)
    Iterator = type(None)
    LabelField = type(None)
