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
import argparse
import logging
from shutil import copyfile

import torch

from pytorch_lightning.utilities.migration.base import pl_legacy_patch
from pytorch_lightning.utilities.migration.migrations import migrate_checkpoint

log = logging.getLogger(__name__)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Upgrade an old checkpoint to the current schema. This will also save a backup of the original file."
        )
    )
    parser.add_argument("--file", help="filepath for a checkpoint to upgrade")

    args = parser.parse_args()

    log.info("Creating a backup of the existing checkpoint file before overwriting in the upgrade process.")
    copyfile(args.file, args.file + ".bak")
    with pl_legacy_patch():
        checkpoint = torch.load(args.file)
    migrate_checkpoint(checkpoint)
    torch.save(checkpoint, args.file)
