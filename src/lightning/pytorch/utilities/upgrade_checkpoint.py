# Copyright The Lightning AI team.
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

from lightning.pytorch.utilities.migration import migrate_checkpoint, pl_legacy_patch

_log = logging.getLogger(__name__)


def _upgrade(args: Namespace) -> None:
    path = Path(args.path).absolute()
    extension: str = args.extension if args.extension.startswith(".") else f".{args.extension}"
    files: List[Path] = []

    if not path.exists():
        _log.error(
            f"The path {path} does not exist. Please provide a valid path to a checkpoint file or a directory"
            f" containing checkpoints ending in {extension}."
        )
        exit(1)

    if path.is_file():
        files = [path]
    if path.is_dir():
        files = [Path(p) for p in glob.glob(str(path / "**" / f"*{extension}"), recursive=True)]
    if not files:
        _log.error(
            f"No checkpoint files with extension {extension} were found in {path}."
            f" HINT: Try setting the `--extension` option to specify the right file extension to look for."
        )
        exit(1)

    _log.info("Creating a backup of the existing checkpoint files before overwriting in the upgrade process.")
    for file in files:
        backup_file = file.with_suffix(".bak")
        if backup_file.exists():
            # never overwrite backup files - they are the original, untouched checkpoints
            continue
        copyfile(file, backup_file)

    _log.info("Upgrading checkpoints ...")
    for file in tqdm(files):
        with pl_legacy_patch():
            checkpoint = torch.load(file)
        migrate_checkpoint(checkpoint)
        torch.save(checkpoint, file)

    _log.info("Done.")


def main() -> None:
    parser = ArgumentParser(
        description=(
            "A utility to upgrade old checkpoints to the format of the current Lightning version."
            " This will also save a backup of the original files."
        )
    )
    parser.add_argument("path", type=str, help="Path to a checkpoint file or a directory with checkpoints to upgrade")
    parser.add_argument(
        "--extension",
        "-e",
        type=str,
        default=".ckpt",
        help="The file extension to look for when searching for checkpoint files in a directory.",
    )
    args = parser.parse_args()
    _upgrade(args)


if __name__ == "__main__":
    main()
