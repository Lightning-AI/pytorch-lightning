import pickle
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import torch

from lightning.fabric.utilities.types import _PATH


def unshard_checkpoint(checkpoint_folder: _PATH, output_file: Optional[_PATH] = None) -> None:
    """Converts a sharded checkpoint saved with the `torch.distributed.checkpoint` API into a regular checkpoint that
    can be loaded with `torch.load()`.

    The current implementation assumes that the entire checkpoint fits in CPU memory.

    """
    # TODO: PyTorch version check

    from torch.distributed.checkpoint import FileSystemReader, load_state_dict
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata

    # TODO: Check if these paths exist
    checkpoint_folder = Path(checkpoint_folder)
    output_file = Path(
        output_file if output_file is not None else checkpoint_folder.with_suffix(checkpoint_folder.suffix + ".merged")
    )
    metadata_file = checkpoint_folder / ".metadata"

    with open(metadata_file, "rb") as file:
        metadata: Metadata = pickle.load(file)

    # TODO: Add sequential save to avoid storing the entire checkpoint in memory
    state_dict = {}
    for tensor_name, metadata in metadata.state_dict_metadata.items():
        if isinstance(metadata, BytesStorageMetadata):  # TODO: What does this represent?
            continue
        state_dict[tensor_name] = torch.empty(
            size=metadata.size, dtype=metadata.properties.dtype, device=torch.device("cpu")
        )

    reader = FileSystemReader(checkpoint_folder)
    load_state_dict(state_dict=state_dict, storage_reader=reader, no_dist=True)
    torch.save(state_dict, output_file)


def main() -> None:
    parser = ArgumentParser(description=("Merges a sharded checkpoint into a single file."))
    parser.add_argument(
        "checkpoint_folder",
        type=str,
        help=(
            "Path to a checkpoint folder, containing the sharded checkpoint files saved using the"
            " `torch.distributed.checkpoint` API."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help=(
            "Path to the file where the merged checkpoint should be saved. The file should not already exist."
            " If no path is provided, the file will be saved next to the input checkpoint folder with the same name"
            " and a '.merged' suffix."
        ),
    )
    args = parser.parse_args()
    unshard_checkpoint(
        checkpoint_folder=args.checkpoint_folder,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
