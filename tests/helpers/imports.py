from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8

if _TORCH_GREATER_EQUAL_1_8:
    from torchtext.legacy.data import Batch, Dataset, Example, Field, Iterator, LabelField
else:
    from torchtext.data import Batch, Dataset, Example, Field, Iterator, LabelField  # noqa: F401
