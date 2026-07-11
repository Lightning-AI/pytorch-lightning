import logging
import sys
from argparse import ArgumentParser, Namespace

from lightning.fabric.utilities.cloud_io import (
    _atomic_save,
    _checkpoint_join,
    _is_checkpoint_dir,
    _resolve_path,
    get_filesystem,
)
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
    checkpoint_fs = get_filesystem(checkpoint_folder)
    if not checkpoint_fs.exists(str(checkpoint_folder)):
        _log.error(f"The provided checkpoint folder does not exist: {checkpoint_folder}")
        sys.exit(1)
    if not _is_checkpoint_dir(checkpoint_folder):
        _log.error(
            f"The provided checkpoint path must be a folder, containing the checkpoint shards: {checkpoint_folder}"
        )
        sys.exit(1)
    if not checkpoint_fs.isfile(str(_checkpoint_join(checkpoint_folder, _METADATA_FILENAME))):
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
    print(f"{config=}")
    # exit(0)
    checkpoint = _load_distributed_checkpoint(config.checkpoint_folder)
    _atomic_save(checkpoint, config.output_file)
