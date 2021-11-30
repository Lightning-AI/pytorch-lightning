from pytorch_lightning.utilities.imports import _TORCH_GREATER_EQUAL_1_8, _TORCHTEXT_AVAILABLE

if _TORCHTEXT_AVAILABLE:
    if _TORCH_GREATER_EQUAL_1_8:
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
