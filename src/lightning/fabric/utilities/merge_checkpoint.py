import logging
from argparse import ArgumentParser
from pathlib import Path

import torch

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1
from lightning.fabric.utilities.load import _load_distributed_checkpoint

_log = logging.getLogger(__name__)


def _cli() -> None:
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

    if not _TORCH_GREATER_EQUAL_2_1:
        _log.error("Processing distributed checkpoints requires PyTorch >= 2.1.")
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

    if args.output_file is None:
        output_file = checkpoint_folder.with_suffix(checkpoint_folder.suffix + ".merged")
    else:
        output_file = Path(args.output_file)
    if output_file.exists():
        _log.error(
            "The path for the merged checkpoint already exists. Choose a different path by providing"
            f" `--output_file` or move/delete the file first: {output_file}"
        )
        exit(1)

    checkpoint = _load_distributed_checkpoint(checkpoint_folder)
    torch.save(checkpoint, output_file)


if __name__ == "__main__":
    _cli()
