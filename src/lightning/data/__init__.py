import sys

from lightning_utilities.core.imports import RequirementCache

_LIGHTNING_DATA_AVAILABLE = RequirementCache("lightning_data")
_LIGHTNING_SDK_AVAILABLE = RequirementCache("lightning_sdk")

if _LIGHTNING_DATA_AVAILABLE:
    import lightning_data

    # Enable resolution at least for lower data namespace
    sys.modules["lightning.data"] = lightning_data

    from lightning_data.processing.functions import map, optimize, walk
    from lightning_data.streaming.combined import CombinedStreamingDataset
    from lightning_data.streaming.dataloader import StreamingDataLoader
    from lightning_data.streaming.dataset import StreamingDataset

else:
    # TODO: Delete all the code when everything is moved to lightning_data
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

# TODO: Move this to lightning_data
if _LIGHTNING_SDK_AVAILABLE:
    from lightning_sdk import Machine  # noqa: F401

    __all__.append("Machine")
