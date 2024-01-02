import logging
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from lightning.fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_1

_log = logging.getLogger(__name__)
_METADATA_FILENAME = "meta.pt"


def load_distributed_checkpoint(checkpoint_folder: Path) -> Dict[str, Any]:
    """Loads a sharded checkpoint saved with the `torch.distributed.checkpoint` into a full state dict.

    The current implementation assumes that the entire checkpoint fits in CPU memory.

    """
    if not _TORCH_GREATER_EQUAL_2_1:
        raise ImportError("Processing distributed checkpoints requires PyTorch >= 2.1.")

    from torch.distributed.checkpoint import FileSystemReader, load_state_dict
    from torch.distributed.checkpoint.metadata import BytesStorageMetadata, TensorStorageMetadata

    reader = FileSystemReader(checkpoint_folder)
    metadata = reader.read_metadata()

    # TODO: Add sequential save to avoid storing the entire checkpoint in memory
    checkpoint = {}
    for tensor_name, sd_metadata in metadata.state_dict_metadata.items():
        if isinstance(sd_metadata, BytesStorageMetadata):
            checkpoint[tensor_name] = "<bytes_io>"
        elif isinstance(sd_metadata, TensorStorageMetadata):
            checkpoint[tensor_name] = torch.empty(
                size=sd_metadata.size,
                dtype=sd_metadata.properties.dtype,
                device=torch.device("cpu"),
                memory_format=sd_metadata.properties.memory_format,
                layout=sd_metadata.properties.layout,
                requires_grad=sd_metadata.properties.requires_grad,
                pin_memory=sd_metadata.properties.pin_memory,
            )

    load_state_dict(state_dict=checkpoint, storage_reader=reader, no_dist=True)
    checkpoint = _unflatten_dict(checkpoint, key_map=metadata.planner_data)

    # This is the extra file saved by Fabric, with user data separate from weights and optimizer states
    extra_file = checkpoint_folder / _METADATA_FILENAME
    extra = torch.load(extra_file, map_location="cpu") if extra_file.is_file() else {}
    checkpoint.update(extra)

    return checkpoint


def _unflatten_dict(checkpoint: Dict[str, Any], key_map: Dict[str, Tuple[str, ...]]) -> Dict[str, Any]:
    """Converts the flat dictionary with keys 'x.y.z...' to a nested dictionary using the provided key map.

    Args:
        checkpoint: The flat checkpoint dictionary.
        key_map: A dictionary that maps the keys in flattened format 'x.y.z...' to a tuple representing
            the index path into the nested dictonary that this function should construct.

    Example:
        {
            'model.layer.weight': ('model', 'layer.weight'),
            'optimizer.state.layer.weight.step': ('optimizer', 'state', 'layer.weight', 'step'),
            'optimizer.state.layer.weight.exp_avg': ('optimizer', 'state', 'layer.weight', 'exp_avg'),
            'optimizer.state.layer.weight.exp_avg_sq': ('optimizer', 'state', 'layer.weight', 'exp_avg_sq'),
            'optimizer.param_groups': ('optimizer', 'param_groups')
        }

    """
    converted = {}
    for flat_key in checkpoint:
        key_path = key_map[flat_key]
        _set_nested_dict_value(converted, key_path, checkpoint[flat_key])
    return converted


def _set_nested_dict_value(nested_dict: Dict[str, Any], key_path: Tuple[str, ...], value: Any) -> None:
    result = nested_dict
    for key in key_path[:-1]:
        if key not in result:
            result[key] = {}
        result = result[key]
    result[key_path[-1]] = value


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

    checkpoint = load_distributed_checkpoint(checkpoint_folder)
    torch.save(checkpoint, output_file)


if __name__ == "__main__":
    main()
