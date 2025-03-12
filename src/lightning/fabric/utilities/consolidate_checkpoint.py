import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch

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
        exit(1)

    checkpoint_folder = Path(args.checkpoint_folder)
    if not checkpoint_folder.exists():
        _log.error(f"The provided checkpoint folder does not exist: {checkpoint_folder}")
        exit(1)
    if not checkpoint_folder.is_dir():
        _log.error(
            f"The provided checkpoint path must be a folder, containing the checkpoint shards: {checkpoint_folder}"
        )
        exit(1)
    if not (checkpoint_folder / _METADATA_FILENAME).is_file():
        _log.error(
            "Only FSDP-sharded checkpoints saved with Lightning are supported for consolidation. The provided folder"
            f" is not in that format: {checkpoint_folder}"
        )
        exit(1)

    if args.output_file is None:
        output_file = checkpoint_folder.with_suffix(checkpoint_folder.suffix + ".consolidated")
    else:
        output_file = Path(args.output_file)
    if output_file.exists():
        _log.error(
            "The path for the converted checkpoint already exists. Choose a different path by providing"
            f" `--output_file` or move/delete the file first: {output_file}"
        )
        exit(1)

    return Namespace(checkpoint_folder=checkpoint_folder, output_file=output_file)


if __name__ == "__main__":
    args = _parse_cli_args()
    config = _process_cli_args(args)
    checkpoint = _load_distributed_checkpoint(config.checkpoint_folder)
    torch.save(checkpoint, config.output_file)
