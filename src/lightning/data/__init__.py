import sys

from lightning_utilities.core.imports import RequirementCache

_LIDATA_AVAILABLE = RequirementCache("litdata")

if not _LIDATA_AVAILABLE:
    raise ModuleNotFoundError("Please, run `pip install litdata`")  # E111

import litdata  # noqa: E402

# Enable resolution at least for lower data namespace
sys.modules["lightning.data"] = litdata

from litdata.processing.functions import map, optimize, walk  # noqa: E402
from litdata.streaming.combined import CombinedStreamingDataset  # noqa: E402
from litdata.streaming.dataloader import StreamingDataLoader  # noqa: E402
from litdata.streaming.dataset import StreamingDataset  # noqa: E402

__all__ = [
    "LightningDataset",
    "StreamingDataset",
    "CombinedStreamingDataset",
    "StreamingDataLoader",
    "LightningIterableDataset",
    "map",
    "optimize",
    "walk",
]

# TODO: Move this to litdata
if RequirementCache("lightning_sdk"):
    from lightning_sdk import Machine  # noqa: F401

    __all__.append("Machine")
