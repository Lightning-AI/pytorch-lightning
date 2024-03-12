import sys

from lightning_utilities.core.imports import RequirementCache

__all__ = []

if not RequirementCache("litdata"):
    raise ModuleNotFoundError("Please, run `pip install litdata`")  # E111

else:
    import litdata

    # Enable resolution at least for lower data namespace
    sys.modules["lightning.data"] = litdata

    from litdata.processing.functions import map, optimize, walk
    from litdata.streaming.combined import CombinedStreamingDataset
    from litdata.streaming.dataloader import StreamingDataLoader
    from litdata.streaming.dataset import StreamingDataset

    __all__ += [
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
    from lightning_sdk import Machine

    __all__ += ["Machine"]
