# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import glob
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path
from shutil import copyfile
from typing import List

import torch
from tqdm import tqdm

from lightning_lite.utilities.types import _PATH
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.utilities.migration import pl_legacy_patch

KEYS_MAPPING = {
    "checkpoint_callback_best_model_score": (ModelCheckpoint, "best_model_score"),
    "checkpoint_callback_best_model_path": (ModelCheckpoint, "best_model_path"),
    "checkpoint_callback_best": (ModelCheckpoint, "best_model_score"),
    "early_stop_callback_wait": (EarlyStopping, "wait_count"),
    "early_stop_callback_patience": (EarlyStopping, "patience"),
}

log = logging.getLogger(__name__)


def upgrade_checkpoint(filepath: _PATH) -> None:
    checkpoint = torch.load(filepath)
    checkpoint["callbacks"] = checkpoint.get("callbacks") or {}

    for key, new_path in KEYS_MAPPING.items():
        if key in checkpoint:
            value = checkpoint[key]
            callback_type, callback_key = new_path
            checkpoint["callbacks"][callback_type] = checkpoint["callbacks"].get(callback_type) or {}
            checkpoint["callbacks"][callback_type][callback_key] = value
            del checkpoint[key]

    torch.save(checkpoint, filepath)


def main(args: Namespace) -> None:
    path = Path(args.path).absolute()
    extension: str = args.extension if args.extension.startswith(".") else f".{args.extension}"
    files: List[Path] = []

    if not path.exists():
        log.error(
            f"The path {path} does not exist. Please provide a valid path to a checkpoint file or a directory"
            " containing checkpoints."
        )
        exit(1)

    if path.is_file():
        files = [path]
    if path.is_dir():
        files = [Path(p) for p in glob.glob(str(path / "**" / f"*{extension}"), recursive=True)]
    if not files:
        log.error(
            f"No checkpoint files with extension {extension} were found in {path}."
            f" HINT: Try setting the `--extension` option to specify the right file extension to look for."
        )
        exit(1)

    log.info("Creating a backup of the existing checkpoint files before overwriting in the upgrade process.")
    for file in files:
        backup_file = file.with_suffix(".bak")
        if backup_file.exists():
            # never overwrite backup files - they are the original, untouched checkpoints
            continue
        copyfile(file, backup_file)

    log.info("Upgrading checkpionts ...")
    for file in tqdm(files):
        with pl_legacy_patch():
            upgrade_checkpoint(file)


if __name__ == "__main__":
    parser = ArgumentParser(
        description=(
            "A utility to upgrade old checkpoints to the format of the current Lightning version."
            " By default, this will also save a backup of the original file."
        )
    )
    parser.add_argument("path", type=str, help="Path to a checkpoint file or a directory with checkpoints to upgrade")
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="If the specified path is a directory, recursively search for checkpoint files to upgrade",
    )
    parser.add_argument(
        "--extension",
        "-e",
        type=str,
        default=".ckpt",
        help="The file extension to look for when searching for checkpoint files in a directory.",
    )
    args = parser.parse_args()
    main(args)
