import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Union

from lightning.fabric.utilities.cloud_io import (
    _atomic_save,
    _checkpoint_join,
    _is_checkpoint_dir,
    _resolve_path,
)
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.fabric.utilities.load import _METADATA_FILENAME, _load_distributed_checkpoint

_log = logging.getLogger(__name__)


def _parse_cli_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Converts a distributed/sharded checkpoint into a single file that can be loaded with `torch.load()`."
            " Only supports FSDP sharded checkpoints at the moment."
            " Supports local paths and fsspec URLs (e.g., s3://bucket/path)."
        ),
    )
    parser.add_argument(
        "checkpoint_folder",
        type=str,
        help=(
            "Path to a checkpoint folder, containing the sharded checkpoint files saved using the"
            " `torch.distributed.checkpoint` API. Can be a local path or an fsspec URL."
        ),
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help=(
            "Path to the file where the converted checkpoint should be saved. The file should not already exist."
            " If no path is provided, the file will be saved next to the input checkpoint folder with the same name"
            " and a '.consolidated' suffix. Can be a local path or an fsspec URL."
        ),
    )
    return parser.parse_args()


def _process_cli_args(args: Namespace) -> Namespace:
    if not _TORCH_GREATER_EQUAL_2_3:
        _log.error("Processing distributed checkpoints requires PyTorch >= 2.3.")
        sys.exit(1)

    checkpoint_folder = _resolve_path(args.checkpoint_folder)
    
    # Validate checkpoint folder exists and is a directory
    if not _is_checkpoint_dir(checkpoint_folder):
        _log.error(f"The provided checkpoint folder does not exist or is not a directory: {checkpoint_folder}")
        sys.exit(1)
    
    # Check for metadata file
    metadata_file = _checkpoint_join(checkpoint_folder, _METADATA_FILENAME)
    if isinstance(metadata_file, Path):
        has_metadata = metadata_file.is_file()
    else:
        from lightning.fabric.utilities.cloud_io import get_filesystem
        has_metadata = get_filesystem(metadata_file).exists(metadata_file)
    
    if not has_metadata:
        _log.error(
            "Only FSDP-sharded checkpoints saved with Lightning are supported for consolidation. The provided folder"
            f" is not in that format: {checkpoint_folder}"
        )
        sys.exit(1)

    if args.output_file is None:
        if isinstance(checkpoint_folder, Path):
            output_file = checkpoint_folder.with_suffix(checkpoint_folder.suffix + ".consolidated")
        else:
            output_file = checkpoint_folder.rstrip("/") + ".consolidated"
    else:
        output_file = _resolve_path(args.output_file)
    
    # Check if output file already exists
    if isinstance(output_file, Path):
        if output_file.exists():
            _log.error(
                "The path for the converted checkpoint already exists. Choose a different path by providing"
                f" `--output_file` or move/delete the file first: {output_file}"
            )
            sys.exit(1)
    else:
        from lightning.fabric.utilities.cloud_io import get_filesystem
        if get_filesystem(output_file).exists(output_file):
            _log.error(
                "The path for the converted checkpoint already exists. Choose a different path by providing"
                f" `--output_file` or move/delete the file first: {output_file}"
            )
            sys.exit(1)

    return Namespace(checkpoint_folder=checkpoint_folder, output_file=output_file)


if __name__ == "__main__":
    args = _parse_cli_args()
    config = _process_cli_args(args)
    checkpoint = _load_distributed_checkpoint(config.checkpoint_folder)
    _atomic_save(checkpoint, config.output_file)
