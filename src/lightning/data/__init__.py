import sys

from lightning_utilities.core.imports import RequirementCache

_LITDATA_AVAILABLE = RequirementCache("litdata")
_LIGHTNING_SDK_AVAILABLE = RequirementCache("lightning_sdk")

if _LITDATA_AVAILABLE:
    import litdata

    # Enable resolution at least for lower data namespace
    sys.modules["lightning.data"] = litdata

    from litdata.processing.functions import map, optimize, walk
    from litdata.streaming.combined import CombinedStreamingDataset
    from litdata.streaming.dataloader import StreamingDataLoader
    from litdata.streaming.dataset import StreamingDataset

else:
    # TODO: Delete all the code when everything is moved to litdata
    from lightning.data.processing.functions import map, optimize, walk
    from lightning.data.streaming.combined import CombinedStreamingDataset
    from lightning.data.streaming.dataloader import StreamingDataLoader
    from lightning.data.streaming.dataset import StreamingDataset

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
if _LIGHTNING_SDK_AVAILABLE:
    from lightning_sdk import Machine  # noqa: F401

    __all__.append("Machine")
