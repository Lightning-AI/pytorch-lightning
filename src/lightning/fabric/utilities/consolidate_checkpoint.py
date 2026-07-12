import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

from lightning.fabric.utilities.cloud_io import _checkpoint_join, _resolve_path, get_filesystem
from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_3
from lightning.fabric.utilities.load import _METADATA_FILENAME, _load_distributed_checkpoint

_log = logging.getLogger(__name__)


def _parse_cli_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Converts a distributed/sharded checkpoint into a single file that can be loaded with `torch.load()`."
            " Only supports FSDP sharded checkpoints at the moment."
        ),
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
            "Path to the file where the converted checkpoint should be saved. The file should not already exist."
            " If no path is provided, the file will be saved next to the input checkpoint folder with the same name"
            " and a '.consolidated' suffix."
        ),
    )
    return parser.parse_args()


def _process_cli_args(args: Namespace) -> Namespace:
    if not _TORCH_GREATER_EQUAL_2_3:
        _log.error("Processing distributed checkpoints requires PyTorch >= 2.3.")
        sys.exit(1)

    checkpoint_folder = _resolve_path(args.checkpoint_folder)
    if isinstance(checkpoint_folder, Path):
        if not checkpoint_folder.exists():
            _log.error(f"The provided checkpoint folder does not exist: {checkpoint_folder}")
            sys.exit(1)
        if not checkpoint_folder.is_dir():
            _log.error(
                f"The provided checkpoint path must be a folder, containing the checkpoint shards: {checkpoint_folder}"
            )
            sys.exit(1)

    # Directories are virtual on remote/object storage, where `isdir()`/`exists()` can be unreliable. The presence
    # of the metadata file that Lightning writes alongside the checkpoint shards is a more robust signal there
    # (see `_is_sharded_checkpoint` in `lightning.fabric.strategies.fsdp`).
    if not get_filesystem(checkpoint_folder).isfile(str(_checkpoint_join(checkpoint_folder, _METADATA_FILENAME))):
        _log.error(
            "Only FSDP-sharded checkpoints saved with Lightning are supported for consolidation. The provided folder"
            f" is not in that format: {checkpoint_folder}"
        )
        sys.exit(1)

    if args.output_file is None:
        output_file = _resolve_path(str(checkpoint_folder).rstrip("/") + ".consolidated")
    else:
        output_file = _resolve_path(args.output_file)
    if get_filesystem(output_file).exists(str(output_file)):
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
    # TODO: replace `torch.save` with `_atomic_save` once #21799 lands.
    # `_atomic_save` can silently succeed without writing anything on permission errors.
    torch.save(checkpoint, config.output_file)
