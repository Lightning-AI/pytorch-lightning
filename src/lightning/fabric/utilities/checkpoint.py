import pickle
from argparse import ArgumentParser
from pathlib import Path

import torch

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1


def unshard_checkpoint(checkpoint_folder: Path, output_file: Path) -> None:
    """Converts a sharded checkpoint saved with the `torch.distributed.checkpoint` API into a regular checkpoint that
    can be loaded with `torch.load()`.

    The current implementation assumes that the entire checkpoint fits in CPU memory.

    """
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ImportError("Processing distributed checkpoints requires PyTorch >= 2.1.")

    from torch.distributed.checkpoint import FileSystemReader, load_state_dict
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, Metadata, TensorStorageMetadata

    metadata_file = checkpoint_folder / ".metadata"
    with open(metadata_file, "rb") as file:
        metadata: Metadata = pickle.load(file)

    # TODO: Add sequential save to avoid storing the entire checkpoint in memory
    state_dict = {}
    for tensor_name, metadata in metadata.state_dict_metadata.items():
        if isinstance(metadata, BytesStorageMetadata):  # TODO: What does this represent?
            continue
        elif isinstance(metadata, TensorStorageMetadata):
            state_dict[tensor_name] = torch.empty(
                size=metadata.size,
                dtype=metadata.properties.dtype,
                device=torch.device("cpu"),
                memory_format=metadata.properties.memory_format,
                layout=metadata.properties.layout,
                requires_grad=metadata.properties.requires_grad,
                pin_memory=metadata.properties.pin_memory,
            )

    load_state_dict(state_dict=state_dict, storage_reader=FileSystemReader(checkpoint_folder), no_dist=True)
    torch.save(state_dict, output_file)


def main() -> None:
    parser = ArgumentParser(
        description="Merges a distributed/sharded checkpoint into a single file that can be loaded with `torch.load()`."
    )
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

    checkpoint_folder = Path(args.checkpoint_folder)
    if not checkpoint_folder.exists():
        raise FileNotFoundError(f"The provided checkpoint folder does not exist: {checkpoint_folder}")
    if not checkpoint_folder.is_dir():
        raise FileNotFoundError(
            f"The provided checkpoint path must be a folder, containing the checkpoint shards: {checkpoint_folder}"
        )
    if args.output_file is None:
        output_file = checkpoint_folder.with_suffix(checkpoint_folder.suffix + ".merged")
    else:
        output_file = Path(args.output_file)
    if output_file.exists():
        raise FileExistsError(
            "The path for the merged checkpoint already exists. Choose a different path by providing"
            f" `--output_file` or move/delete the file first: {output_file}"
        )

    unshard_checkpoint(checkpoint_folder=checkpoint_folder, output_file=output_file)


if __name__ == "__main__":
    main()
